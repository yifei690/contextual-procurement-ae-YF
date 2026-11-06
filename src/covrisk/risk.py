"""Finite W2 transport ambiguity and a *joint* spectral-risk adversary.

The same q governs the expectation and every CVaR component. Prices/costs
may be negative; dual a and b are unrestricted. SciPy/HiGHS is sufficient.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class RiskSpec:
    rho: float = 0.3
    alpha: tuple = (0.90, 0.95)
    beta: tuple = (0.5, 0.5)

    def validate(self):
        if not 0 <= self.rho < 1:
            raise ValueError("rho must lie in [0,1)")
        if len(self.alpha) != len(self.beta) or not self.alpha:
            raise ValueError("risk levels and weights must match")
        if not all(0 < a < 1 for a in self.alpha):
            raise ValueError("CVaR levels must lie in (0,1)")
        if min(self.beta) < 0 or not np.isclose(sum(self.beta), 1):
            raise ValueError("CVaR weights must be a probability vector")


def validate_ambiguity(weights, distance, radius):
    w, d = np.asarray(weights, float), np.asarray(distance, float)
    if d.ndim != 2 or d.shape[0] != len(w) or d.shape[1] < 1:
        raise ValueError("inconsistent transport dimensions")
    if not np.isfinite(d).all() or not np.isfinite(w).all():
        raise ValueError("nonfinite ambiguity parameters")
    if min(w) < 0 or not np.isclose(w.sum(), 1) or d.min() < -1e-10:
        raise ValueError("invalid probability/distance")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("invalid radius")
    if w @ d.min(axis=1) > radius**2 + 1e-10:
        raise ValueError("empty transport ambiguity set")
    return w, d


def adversary(cost, weights, distance, radius, risk=RiskSpec()):
    """Maximize combined expectation/CVaR; return a supporting risk cut."""
    risk.validate()
    w, d = validate_ambiguity(weights, distance, radius)
    c = np.asarray(cost, float)
    n, j = d.shape
    k = len(risk.alpha)
    if c.shape != (j,) or not np.isfinite(c).all():
        raise ValueError("invalid scenario costs")
    total = n*j + k*j
    objective = np.r_[(1-risk.rho)*np.tile(c, n),
                      np.concatenate([risk.rho*b*c for b in risk.beta])]
    eq = lil_matrix((n+k, total))
    for i in range(n):
        eq[i, i*j:(i+1)*j] = 1
    for h in range(k):
        eq[n+h, n*j+h*j:n*j+(h+1)*j] = 1
    ub = lil_matrix((1+k*j, total))
    ub[0, :n*j] = d.ravel()
    for h, a in enumerate(risk.alpha):
        for v in range(j):
            row = 1+h*j+v
            ub[row, np.arange(n)*j+v] = -1
            ub[row, n*j+h*j+v] = 1-a
    sol = linprog(-objective, A_ub=ub.tocsr(), b_ub=np.r_[radius**2, np.zeros(k*j)],
                  A_eq=eq.tocsr(), b_eq=np.r_[w, np.ones(k)], bounds=(0, None),
                  method="highs")
    if not sol.success:
        raise RuntimeError(f"risk separation failed: {sol.message}")
    transport = sol.x[:n*j].reshape(n, j)
    q = transport.sum(0)
    v = sol.x[n*j:].reshape(k, j)
    omega = (1-risk.rho)*q + risk.rho*np.asarray(risk.beta) @ v
    return {"value": float(omega@c), "q": q, "v": v, "omega": omega,
            "transport": transport, "transport_used": float((d*transport).sum())}


def dual_value(cost, weights, distance, radius, risk=RiskSpec()):
    """Independent LP implementation of strategy Eq. (23)."""
    risk.validate()
    w, d = validate_ambiguity(weights, distance, radius)
    n, j = d.shape
    k = len(risk.alpha)
    c = np.asarray(cost, float)
    # Variables: a[0:n], lambda[n], b[n+1:n+1+k], mu[k,j].
    offset = n+1+k
    obj = np.r_[w, radius**2, np.ones(k), np.zeros(k*j)]
    ub = lil_matrix((n*j+k*j, len(obj)))
    rhs = np.zeros(n*j+k*j)
    for i in range(n):
        for v in range(j):
            row = i*j+v
            ub[row, i], ub[row, n] = -1, -d[i, v]
            for h in range(k):
                ub[row, offset+h*j+v] = 1
            rhs[row] = -(1-risk.rho)*c[v]
    for h in range(k):
        for v in range(j):
            row = n*j+h*j+v
            ub[row, n+1+h] = -1
            ub[row, offset+h*j+v] = -(1-risk.alpha[h])
            rhs[row] = -risk.rho*risk.beta[h]*c[v]
    bounds = [(None, None)]*n+[(0, None)]+[(None, None)]*k+[(0, None)]*(k*j)
    sol = linprog(obj, A_ub=ub.tocsr(), b_ub=rhs, bounds=bounds, method="highs")
    if not sol.success:
        raise RuntimeError(f"dual LP failed: {sol.message}")
    return float(sol.fun)


def empirical_cvar(costs, alpha, weights=None):
    c = np.asarray(costs, float)
    if c.size == 0:
        raise ValueError("empty costs")
    w = np.full(len(c), 1/len(c)) if weights is None else np.asarray(weights, float)
    order = np.argsort(c)[::-1]
    taken = np.minimum(w[order], np.maximum(0, 1-alpha-np.r_[0, np.cumsum(w[order])[:-1]]))
    return float(taken @ c[order]/(1-alpha))
