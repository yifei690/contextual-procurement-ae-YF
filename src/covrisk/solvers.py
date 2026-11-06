"""Exact finite-model backends and risk-cut generation with valid bounds."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import vstack
from .canonical import Program, scalar
from .physical import scenario_costs
from .risk import RiskSpec, adversary


@dataclass
class SolveResult:
    x: np.ndarray | None
    objective: float
    bound: float
    gap: float
    status: str
    seconds: float
    backend: str
    residuals: dict


def solve_exact(program, backend="highs", time_limit=120., relative_gap=1e-7, relax=False):
    a, e = program.matrices()
    c = program.c.detach().numpy()
    b, rhs = program.b.detach().numpy(), program.e.detach().numpy()
    n = len(c)
    start = time.perf_counter()
    if backend == "highs":
        integrality = np.zeros(n, int)
        if not relax:
            integrality[program.integers] = 1
        constraints = LinearConstraint(vstack([a, e]), np.r_[np.full(len(b), -np.inf), rhs], np.r_[b, rhs])
        sol = milp(c, integrality=integrality, bounds=Bounds(np.full(n, -np.inf), np.full(n, np.inf)),
                   constraints=constraints, options={"time_limit": time_limit, "mip_rel_gap": relative_gap})
        x = sol.x
        status = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "error"}.get(sol.status, "error")
        value = float(sol.fun) if sol.fun is not None else np.inf
        bound = getattr(sol, "mip_dual_bound", None)
        bound = float(bound) if bound is not None else (value if status == "optimal" else -np.inf)
    elif backend == "gurobi":
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                model.Params.Threads = 1
                model.Params.TimeLimit = time_limit
                model.Params.MIPGap = relative_gap
                model.Params.FeasibilityTol = 1e-8
                types = np.full(n, gp.GRB.CONTINUOUS, dtype="U1")
                if not relax:
                    types[program.integers] = gp.GRB.INTEGER
                z = model.addMVar(n, lb=-gp.GRB.INFINITY, vtype=types, obj=c)
                model.addMConstr(a, z, "<", b)
                if len(rhs):
                    model.addMConstr(e, z, "=", rhs)
                model.optimize()
                x = z.X if model.SolCount else None
                value = float(model.ObjVal) if model.SolCount else np.inf
                status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.INFEASIBLE: "infeasible",
                          gp.GRB.UNBOUNDED: "unbounded", gp.GRB.TIME_LIMIT: "limit"}.get(model.Status, "error")
                bound = float(model.ObjBound) if model.IsMIP else (value if status == "optimal" else -np.inf)
    else:
        raise ValueError("backend must be highs or gurobi")
    residuals = program.residuals(x) if x is not None else {}
    if x is not None and (residuals["equality"] > 1e-5 or residuals["inequality"] > 1e-5 or
                           (not relax and residuals["integrality"] > 1e-5)):
        status = "invalid_incumbent"
    gap = max(0., value-bound)/max(1., abs(value)) if np.isfinite(value) else np.inf
    return SolveResult(x, value, bound, gap, status, time.perf_counter()-start, backend, residuals)


def append_cut(program, omega):
    coefficients = {}
    for weight, cost in zip(omega, program.meta["costs"]):
        for idx, value in cost.items():
            coefficients[idx] = coefficients.get(idx, 0)+float(weight)*value
    coefficients[len(program.c)-1] = scalar(-1)
    cols = np.asarray(list(coefficients), int)
    vals = torch.stack([scalar(v) for v in coefficients.values()])
    return Program(program.c, torch.cat([program.av, vals]), torch.cat([program.b, scalar(0).reshape(1)]),
                   program.ev, program.e, np.r_[program.ar, np.full(len(cols), len(program.b), int)],
                   np.r_[program.ac, cols], program.er, program.ec, program.integers,
                   program.names, program.meta)


def solve_cuts(program, weights, distance, radius, risk=RiskSpec(), backend="highs",
               tolerance=1e-7, max_iterations=300, time_limit=120.):
    if program.meta["formulation"] != "cuts":
        raise ValueError("cut master required")
    start = time.perf_counter()
    # A feasible initial risk cut; separation at zero costs returns a feasible point.
    initial = adversary(np.zeros(distance.shape[1]), weights, distance, radius, risk)
    master = append_cut(program, initial["omega"])
    lower, upper, best, history = -np.inf, np.inf, None, []
    status = "iteration_limit"
    for iteration in range(max_iterations):
        remaining = time_limit-(time.perf_counter()-start)
        if remaining <= 0:
            status = "limit"
            break
        sol = solve_exact(master, backend, remaining, relative_gap=min(1e-8, tolerance/10))
        lower = max(lower, sol.bound)
        if sol.x is None or sol.status == "invalid_incumbent":
            status = sol.status
            break
        separation = adversary(scenario_costs(program, sol.x), weights, distance, radius, risk)
        value = separation["value"]
        if value < upper:
            upper, best = value, sol.x.copy()
        history.append({"iteration": iteration, "lower": lower, "upper": upper,
                        "master_status": sol.status})
        if upper-lower <= tolerance*max(1, abs(upper)):
            status = "optimal"
            break
        master = append_cut(master, separation["omega"])
    result = SolveResult(best, upper, lower, (upper-lower)/max(1, abs(upper)) if np.isfinite(upper) else np.inf,
                         status, time.perf_counter()-start, backend+"_cuts", program.residuals(best) if best is not None else {})
    return result, history
