"""Causal receding-horizon execution; actual prices are used only at settlement."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from .canonical import Builder
from .differentiable import solve_differentiable, BarrierSettings
from .distributions import ParameterPack
from .physical import Plant, build_procurement
from .risk import RiskSpec
from .solvers import solve_exact
from .tree import ScenarioTree


class ReplayFailure(RuntimeError):
    pass


@dataclass
class ReplayResult:
    cost: torch.Tensor
    trajectory: list
    solver_records: list
    p: torch.Tensor
    seconds: float


def conditioned_remainder(pack, observed_past, known_da, start, scale, bandwidth=1.):
    """Update using past metered L/G and already-cleared DA, never final IM.

    No future observations are accepted by this API. A fixed differentiable
    likelihood updates the scenario weights. Physical paths are conditioned by
    these weights; topology is fixed for each remaining horizon.
    """
    if observed_past.shape[0] != start or observed_past.shape[-1] != 2:
        raise ValueError("only completed load/PV observations are permitted")
    paths = pack.paths[:, start:].clone()
    j, remaining, _ = paths.shape
    weights = pack.weights
    if start:
        innovation = (pack.paths[:, :start, :2]-observed_past[None])/scale[:2]
        # A horizon-average likelihood prevents unintended exponential collapse.
        log_likelihood = -.5*innovation.square().mean((1, 2))/bandwidth**2
        weights = torch.softmax(torch.log(pack.weights)+log_likelihood, 0)
        weights = .95*weights+.05/j
    # Cleared DA is known at operation time; it never enters the DA solve above.
    paths = torch.cat([paths[..., :2], known_da[start:][None, :, None].expand(j, -1, -1), paths[..., 3:]], -1)
    # Retain the original future reveal schedule. Rebuilding a regular tree
    # after truncation could reveal a split before its L/G histories differ.
    post = pack.tree.post[:, start:].copy()
    pre = np.zeros_like(post)
    pre[:, 1:] = post[:, :-1]
    tree = ScenarioTree(pre, post)
    tree.validate()
    paths = tree.project(paths)
    standardized = paths/scale
    distance = (standardized[:, None]-standardized[None, :]).square().mean((2, 3))
    return ParameterPack(paths, weights, distance, pack.radius, tree, 1/weights.square().sum(), pack.source_ids)


def settle_physical(load, pv, c, d, plant, smooth, settings):
    """Metered balancing uses physical priorities; final prices are NOT inputs."""
    raw = load-pv+c-d
    if not smooth:
        ell = torch.clamp(raw-plant.import_max, min=0)
        k = torch.clamp(-plant.export_max-raw, min=0)
        if float(ell) > plant.flexible_fraction*float(load)+1e-7 or float(k) > float(pv)+1e-7:
            raise ReplayFailure("realized critical load cannot be served by the precommitted action")
        return raw+k-ell, k, ell
    builder = Builder()
    g = builder.var("grid", -plant.export_max, plant.import_max)
    k = builder.var("curtail", 0, 0 if float(pv) == 0 else pv, objective=.01)
    ell = builder.var("unserved", 0, plant.flexible_fraction*load, objective=10.)
    builder.eq({g: 1, k: -1, ell: 1}, raw)
    solution, _ = solve_differentiable(builder.finish(), settings)
    return solution[g], solution[k], solution[ell]


def replay_day(pack, actual, scale, plant=Plant(), risk=RiskSpec(), smooth=False,
               backend="highs", settings=BarrierSettings(), time_limit=60., stop_optimization_gradient=False):
    """Full day with one DA decision and a new optimization before every hour.

    Runtime predictor conditioning accepts only past physical observations and
    cleared day-ahead prices. State/positions remain on graph during training.
    DA, incremental ID and deviations are each billed once.
    """
    start_time = time.perf_counter()
    j, horizon, _ = pack.paths.shape
    if actual.shape != (horizon, 5):
        raise ValueError("invalid actual trajectory")
    records = []
    def optimize(current_pack, state, fixed_da=None):
        program = build_procurement(current_pack.paths, current_pack.tree, current_pack.weights,
            current_pack.distance, current_pack.radius, plant, risk, initial_energy=state, fixed_da=fixed_da)
        if smooth:
            solution, info = solve_differentiable(program, settings)
            records.append(info)
        else:
            result = solve_exact(program, backend=backend, time_limit=time_limit)
            records.append({"status": result.status, "objective": result.objective, "bound": result.bound,
                            "gap": result.gap, "seconds": result.seconds, **result.residuals})
            if result.x is None or result.status not in ("optimal", "limit"):
                raise ReplayFailure(f"no verified executable incumbent: {result.status}")
            solution = torch.tensor(result.x, dtype=torch.float64)
        if stop_optimization_gradient:
            solution = solution.detach()
        return program, solution
    state = torch.tensor(plant.initial_energy, dtype=torch.float64)
    day_program, day_solution = optimize(pack, state)
    p = day_solution[day_program.meta["p"]]
    # This operation is settlement, occurring after the DA decision was fixed.
    total = (actual[:, 2]*p).sum()/1000
    trajectory = []
    for t in range(horizon):
        current = conditioned_remainder(pack, actual[:t, :2], actual[:, 2], t, scale)
        program, solution = optimize(current, state, p[t:])
        refs = program.meta["refs"]
        c, d, x = (solution[refs[key][0, 0]] for key in ("c", "d", "x"))
        load, pv, da_price, id_price, im_price = actual[t]
        g, k, ell = settle_physical(load, pv, c, d, plant, smooth, settings)
        imbalance = g-p[t]-x
        new_state = state+plant.eta_c*c-d/plant.eta_d
        da_cost = da_price*p[t]
        id_cost = id_price*x
        im_cost = im_price*imbalance
        degradation = plant.degradation_eur_mwh*(c+d)
        curtail_cost = plant.curtailment_eur_mwh*k
        shortage_cost = plant.voll_eur_mwh*ell
        incremental = id_cost+im_cost+degradation+curtail_cost+shortage_cost
        total = total+incremental/1000
        # The stored numeric trace is not used for backpropagation.
        values = {"load": load, "pv": pv, "p": p[t], "id_increment": x, "imbalance": imbalance,
                  "grid": g, "charge": c, "discharge": d, "soc_start": state, "soc_end": new_state,
                  "curtailment": k, "unserved": ell, "da_price": da_price, "id_price": id_price,
                  "im_price": im_price, "da_cost": da_cost, "id_cost": id_cost, "im_cost": im_cost,
                  "degradation": degradation, "curtail_cost": curtail_cost, "shortage_cost": shortage_cost,
                  "total_cost": da_cost+incremental}
        trajectory.append({"hour_index": t, **{key: float(value.detach()) for key, value in values.items()}})
        state = new_state
    return ReplayResult(total, trajectory, records, p, time.perf_counter()-start_time)
