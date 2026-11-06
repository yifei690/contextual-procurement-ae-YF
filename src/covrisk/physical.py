"""Shared-node multistage procurement, storage, and exact DRO dual.

Variables use MW/MWh and costs use thousands of EUR to condition the KKT
system. Single-price ID/imbalance use signed positions: their positive and
negative parts give mutually exclusive trade directions exactly, even at
negative prices. Dual-price contracts are intentionally rejected by schema.
"""
from dataclasses import asdict, dataclass
import numpy as np
import torch
from .canonical import Builder, add_to, scalar
from .risk import RiskSpec, validate_ambiguity
from .tree import ScenarioTree

FIELDS = ("load", "pv", "day_ahead", "intraday", "imbalance")


@dataclass(frozen=True)
class Plant:
    import_max: float = 2.5
    export_max: float = 1.0
    da_max: float = 2.5
    id_max: float = 1.0
    charge_max: float = 0.5
    discharge_max: float = 0.5
    energy_min: float = 0.2
    energy_max: float = 2.0
    initial_energy: float = 1.0
    terminal_energy: float | None = 1.0
    eta_c: float = 0.95
    eta_d: float = 0.95
    flexible_fraction: float = 0.1
    degradation_eur_mwh: float = 10.0
    curtailment_eur_mwh: float = 0.0
    voll_eur_mwh: float = 10000.0

    def validate(self):
        if min(self.import_max, self.export_max, self.da_max, self.id_max, self.charge_max, self.discharge_max) < 0:
            raise ValueError("negative equipment/trade limit")
        if not self.energy_min <= self.initial_energy <= self.energy_max:
            raise ValueError("invalid initial SOC")
        if self.terminal_energy is not None and not self.energy_min <= self.terminal_energy <= self.energy_max:
            raise ValueError("invalid terminal SOC")
        if not 0 < self.eta_c <= 1 or not 0 < self.eta_d <= 1 or not 0 <= self.flexible_fraction <= 1:
            raise ValueError("invalid efficiency/flexibility")


def build_procurement(paths, tree, weights, distance, radius, plant=Plant(), risk=RiskSpec(),
                      dt=1.0, initial_energy=None, fixed_da=None, formulation="dual"):
    """Build one problem; exact and training solvers consume identical coefficients.

    Day-ahead p is at the root; x,c,d precede the current interval observation;
    grid/curtailment/unserved load react to physical realization. A single ID
    update for the imminent delivery interval is permitted. Its settlement
    index is uncertain at the decision node. This is an indexed-contract
    abstraction, not a reconstruction of executable order-book trades.
    """
    plant.validate()
    risk.validate()
    if paths.shape[-1] != len(FIELDS):
        raise ValueError("five single-price fields required")
    tree.check_paths(paths)
    j, h, _ = paths.shape
    if bool((paths[..., :2].detach() < -1e-9).any()):
        raise ValueError("negative physical load or PV")
    if initial_energy is None:
        initial_energy = plant.initial_energy
    durations = np.broadcast_to(np.asarray(dt, float), (h,))
    if np.min(durations) <= 0:
        raise ValueError("nonpositive duration")
    builder = Builder()
    refs = {name: np.empty((j, h), int) for name in ("x", "c", "d", "e", "u", "g", "b", "k", "ell")}
    p = []
    for t in range(h):
        if fixed_da is None:
            idx = builder.var(f"p[{t}]", 0, plant.da_max)
        else:
            idx = builder.var(f"p[{t}]")
            builder.eq({idx: 1}, fixed_da[t])
        p.append(idx)
    for t in range(h):
        for node in np.unique(tree.pre[:, t]):
            leaves = np.flatnonzero(tree.pre[:, t] == node)
            first = leaves[0]
            key = f"{t},{node}"
            x = builder.var(f"x[{key}]", -plant.id_max, plant.id_max)
            c = builder.var(f"c[{key}]", 0, plant.charge_max)
            d = builder.var(f"d[{key}]", 0, plant.discharge_max)
            u = builder.var(f"u[{key}]", 0, 1, integer=True)
            if t == h-1 and plant.terminal_energy is not None:
                e = builder.var(f"e[{key}]", plant.terminal_energy, plant.terminal_energy)
            else:
                e = builder.var(f"e[{key}]", plant.energy_min, plant.energy_max)
            for name, idx in (("x", x), ("c", c), ("d", d), ("e", e), ("u", u)):
                refs[name][leaves, t] = idx
            if plant.charge_max:
                builder.le({c: 1, u: -plant.charge_max}, 0)
            if plant.discharge_max:
                builder.le({d: 1, u: plant.discharge_max}, plant.discharge_max)
            balance = {e: 1, c: -plant.eta_c*durations[t], d: durations[t]/plant.eta_d}
            if t:
                balance[refs["e"][first, t-1]] = -1
                builder.eq(balance, 0)
            else:
                builder.eq(balance, initial_energy)
        for node in np.unique(tree.post[:, t]):
            leaves = np.flatnonzero(tree.post[:, t] == node)
            first = leaves[0]
            load, pv = paths[first, t, :2]
            key = f"{t},{node}"
            g = builder.var(f"g[{key}]", -plant.export_max, plant.import_max)
            b = builder.var(f"b[{key}]")
            # At structural night, eliminate k with an equality; no empty log interior.
            k = builder.var(f"k[{key}]", 0, 0 if float(pv.detach()) == 0 else pv)
            ell = builder.var(f"ell[{key}]", 0, 0 if plant.flexible_fraction == 0 else plant.flexible_fraction*load)
            for name, idx in (("g", g), ("b", b), ("k", k), ("ell", ell)):
                refs[name][leaves, t] = idx
            builder.eq({g: 1, p[t]: -1, refs["x"][first, t]: -1, b: -1}, 0)
            builder.eq({g: 1, k: -1, refs["d"][first, t]: 1, ell: 1, refs["c"][first, t]: -1}, load-pv)
    costs = []
    for s in range(j):
        coeff = {}
        for t in range(h):
            scale = durations[t]/1000
            for idx, value in ((p[t], paths[s, t, 2]), (refs["x"][s, t], paths[s, t, 3]),
                               (refs["b"][s, t], paths[s, t, 4]),
                               (refs["c"][s, t], plant.degradation_eur_mwh),
                               (refs["d"][s, t], plant.degradation_eur_mwh),
                               (refs["k"][s, t], plant.curtailment_eur_mwh),
                               (refs["ell"][s, t], plant.voll_eur_mwh)):
                add_to(coeff, idx, scalar(value)*scale)
        costs.append(coeff)
    physical_n = len(builder.names)
    if formulation == "dual":
        validate_ambiguity(weights.detach().numpy(), distance.detach().numpy(), float(scalar(radius)))
        # Explicit path-cost variables avoid copying H-period cost coefficients
        # into every transport/risk inequality. This is an exact extended
        # formulation, and exposes a much sparser KKT system during training.
        cost_variables = []
        for s in range(j):
            cj = builder.var(f"path_cost[{s}]")
            cost_variables.append(cj)
            coeff = dict(costs[s])
            coeff[cj] = -1
            builder.eq(coeff, 0)
        a = [builder.var(f"a[{i}]", objective=wi) for i, wi in enumerate(weights)]
        lam = builder.var("lambda", 0, objective=scalar(radius)**2)
        bs = [builder.var(f"risk_b[{k}]", objective=1) for k in range(len(risk.alpha))]
        mu = [[builder.var(f"mu[{k},{s}]", 0) for s in range(j)] for k in range(len(risk.alpha))]
        for i in range(len(weights)):
            for s in range(j):
                coeff = {cost_variables[s]: 1-risk.rho, a[i]: -1, lam: -distance[i, s]}
                for k in range(len(risk.alpha)):
                    coeff[mu[k][s]] = 1
                builder.le(coeff, 0)
        for k, alpha in enumerate(risk.alpha):
            for s in range(j):
                coeff = {cost_variables[s]: risk.rho*risk.beta[k], bs[k]: -1, mu[k][s]: -(1-alpha)}
                builder.le(coeff, 0)
    elif formulation == "cuts":
        builder.var("risk_epigraph", objective=1)
    else:
        raise ValueError("unknown formulation")
    meta = {"refs": refs, "p": np.asarray(p), "costs": costs, "physical_n": physical_n,
            "plant": asdict(plant), "tree": tree, "dt": durations, "formulation": formulation,
            "cost_unit": "1000 EUR", "contract": "single-price indexed incremental intraday"}
    return builder.finish(meta)


def scenario_costs(program, solution):
    if isinstance(solution, torch.Tensor):
        return torch.stack([sum(value*solution[idx] for idx, value in c.items()) for c in program.meta["costs"]])
    return np.asarray([sum(float(value.detach())*solution[idx] for idx, value in c.items()) for c in program.meta["costs"]])
