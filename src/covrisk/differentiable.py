"""Sparse strongly-convex log-barrier solve with a full implicit adjoint.

No differentiation of integer optimizers, finite differences, or stop-gradient
through the transport/physical coefficients. Forward/backward residuals are
checked; failures raise LayerFailure and must be counted by the trainer.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import splu


class LayerFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class BarrierSettings:
    mu: float = 1e-4
    tau: float = 1e-4
    tolerance: float = 2e-8
    max_iterations: int = 150


class AugmentedFactor:
    def __init__(self, factor, original_size, extra):
        self.factor, self.original_size, self.extra = factor, original_size, extra

    def solve(self, rhs, trans="N"):
        full = self.factor.solve(np.r_[rhs, np.zeros(self.extra)], trans=trans)
        return full[:self.original_size]


class _KKTSystem:
    """Reuse a single solve's CSC structure; rebind coefficients on every solve."""

    def __init__(self, a, e, mu, tau):
        # Eliminating the last block gives mu*I + A' diag(tau/slack^2) A.
        # Only its slack diagonal changes during Newton iteration. Keep this
        # cache local: the next autograd call may have different A/E values.
        n, m = a.shape[1], a.shape[0]
        self.original_size, self.extra, self.tau = n+e.shape[0], m, tau
        self.matrix = sparse.bmat([
            [mu*sparse.eye(n), e.T, a.T],
            [e, None, sparse.csr_matrix((e.shape[0], m))],
            [a, sparse.csr_matrix((m, e.shape[0])), -sparse.eye(m)],
        ], format="csc")
        self.matrix.sort_indices()
        columns = np.repeat(np.arange(self.matrix.shape[1]), np.diff(self.matrix.indptr))
        self.slack_positions = np.flatnonzero(
            (self.matrix.indices == columns) & (columns >= self.original_size))

    def factor(self, slack):
        self.matrix.data[self.slack_positions] = -slack**2/self.tau
        try:
            # Each returned SuperLU factor owns its numeric factorization;
            # later diagonal updates cannot change a saved adjoint solve.
            return AugmentedFactor(splu(self.matrix, permc_spec="COLAMD"),
                                   self.original_size, self.extra)
        except RuntimeError as exc:
            raise LayerFailure("singular KKT system; check redundant equalities/interior") from exc


def barrier_forward(c, a, b, e, rhs, settings):
    n, m = len(c), len(b)
    if settings.mu <= 0 or settings.tau <= 0:
        raise ValueError("positive regularization and barrier required")
    # Relative-interior phase I. Fixed quantities must be equality constrained.
    phase = linprog(np.r_[np.zeros(n), -1.], A_ub=sparse.hstack([a, np.ones((m, 1))]), b_ub=b,
                    A_eq=sparse.hstack([e, sparse.csr_matrix((e.shape[0], 1))]), b_eq=rhs,
                    bounds=[(None, None)]*n+[(None, 1.)], method="highs")
    if not phase.success or phase.x[-1] <= 1e-10:
        raise LayerFailure("continuous model has no verified strict relative interior")
    x = phase.x[:n].copy()
    nu = np.zeros(e.shape[0])
    tau, mu = settings.tau, settings.mu
    last_residual = np.inf
    kkt = _KKTSystem(a, e, mu, tau)
    for iteration in range(settings.max_iterations):
        s = b-a@x
        grad = c+mu*x+tau*(a.T@(1/s))
        r1, r2 = grad+e.T@nu, e@x-rhs
        residual = max(np.max(np.abs(r1), initial=0), np.max(np.abs(r2), initial=0))
        lu = kkt.factor(s)
        if residual < settings.tolerance:
            return x, nu, s, lu, {"iterations": iteration, "kkt_residual": residual,
                                         "minimum_slack": float(s.min()), "barrier_gap": m*tau}
        step = lu.solve(-np.r_[r1, r2])
        dx, dn = step[:n], step[n:]
        adx = a@dx
        positive = adx > 0
        length = min(1., float(np.min(0.995*s[positive]/adx[positive], initial=1.)))
        # Residual line search handles an imperfectly feasible initial solution.
        merit = np.linalg.norm(np.r_[r1, r2])
        accepted = False
        for _ in range(65):
            xx, nn = x+length*dx, nu+length*dn
            ss = b-a@xx
            if np.min(ss) > 0:
                rr = np.r_[c+mu*xx+tau*(a.T@(1/ss))+e.T@nn, e@xx-rhs]
                if np.linalg.norm(rr) <= (1-1e-4*length)*merit:
                    x, nu, accepted = xx, nn, True
                    break
            length *= 0.5
        if not accepted:
            raise LayerFailure(f"barrier line search failed, residual={residual:.3g}")
        last_residual = residual
    raise LayerFailure(f"barrier iteration limit; residual={last_residual:.3g}")


class _ImplicitBarrier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, av, b, ev, rhs, structure, settings, diagnostics):
        ar, ac, er, ec = structure
        n = c.numel()
        a = sparse.coo_matrix((av.detach().numpy(), (ar, ac)), shape=(b.numel(), n)).tocsr()
        e = sparse.coo_matrix((ev.detach().numpy(), (er, ec)), shape=(rhs.numel(), n)).tocsr()
        x, nu, slack, lu, info = barrier_forward(c.detach().numpy(), a, b.detach().numpy(),
                                                       e, rhs.detach().numpy(), settings)
        diagnostics.update(info)
        ctx.payload = (a, e, x, nu, slack, lu, structure, settings, diagnostics)
        return torch.from_numpy(x.copy())

    @staticmethod
    def backward(ctx, output_gradient):
        a, e, x, nu, s, lu, structure, settings, diagnostics = ctx.payload
        ar, ac, er, ec = structure
        right = np.r_[output_gradient.detach().numpy(), np.zeros(len(nu))]
        v = lu.solve(right, trans="T")
        vx, vn = v[:len(x)], v[len(x):]
        applied = np.r_[settings.mu*vx+settings.tau*(a.T@((a@vx)/s**2))+e.T@vn, e@vx]
        res = np.max(np.abs(applied-right), initial=0)/max(1, np.max(np.abs(right), initial=0))
        diagnostics["adjoint_residual"] = float(res)
        if not np.isfinite(v).all() or res > 1e-6:
            raise LayerFailure(f"inaccurate adjoint: {res}")
        avx = a@vx
        gb = settings.tau*avx/s**2
        ga = -settings.tau*(vx[ac]/s[ar]+avx[ar]*x[ac]/s[ar]**2)
        ge = -(nu[er]*vx[ec]+vn[er]*x[ec])
        arrays = (-vx, ga, gb, ge, vn)
        return *(torch.from_numpy(np.asarray(value).copy()) for value in arrays), None, None, None


def solve_differentiable(program, settings=BarrierSettings()):
    diagnostics = {"integer_relaxation": True, "mu": settings.mu, "tau": settings.tau}
    # Nondimensionalize variables before the strong-convexity term. This also
    # handles arbitrarily small but positive daytime PV upper bounds without
    # deleting their derivative or falsely demanding a large absolute slack.
    scales = program.meta.get("variable_scales", torch.ones_like(program.c))
    av = program.av*scales[program.ac]
    ev = program.ev*scales[program.ec]
    a_norm = torch.zeros_like(program.b).scatter_add(0, torch.from_numpy(program.ar), av.square()).sqrt()
    e_norm = torch.zeros_like(program.e).scatter_add(0, torch.from_numpy(program.er), ev.square()).sqrt()
    if bool((a_norm <= 0).any()) or bool((e_norm <= 0).any()):
        raise LayerFailure("zero constraint row requires explicit preprocessing")
    x = _ImplicitBarrier.apply(program.c*scales, av/a_norm[program.ar], program.b/a_norm,
                               ev/e_norm[program.er], program.e/e_norm,
                               (program.ar, program.ac, program.er, program.ec), settings, diagnostics)
    return x*scales, diagnostics
