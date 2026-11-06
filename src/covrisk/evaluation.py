"""Independent physical/cost audit and paired block inference."""
from dataclasses import asdict
import numpy as np
import pandas as pd
from .physical import Plant
from .risk import empirical_cvar


def audit_trajectory(rows, plant=Plant(), tolerance=2e-5):
    f = pd.DataFrame(rows)
    if len(f) == 0:
        return {"passed": False, "reason": "empty trajectory"}
    # Deliberately recomputed from primitive quantities, not solver objectives.
    cost = f.da_price*f.p+f.id_price*f.id_increment+f.im_price*(f.grid-f.p-f.id_increment)
    cost += plant.degradation_eur_mwh*(f.charge+f.discharge)+plant.curtailment_eur_mwh*f.curtailment+plant.voll_eur_mwh*f.unserved
    violation = {
        "balance": np.abs(f.grid+f.pv-f.curtailment+f.discharge+f.unserved-f.load-f.charge).max(),
        "soc_equation": np.abs(f.soc_end-f.soc_start-plant.eta_c*f.charge+f.discharge/plant.eta_d).max(),
        "soc_continuity": np.abs(f.soc_start.to_numpy()[1:]-f.soc_end.to_numpy()[:-1]).max(initial=0),
        "soc_initial": abs(f.soc_start.iloc[0]-plant.initial_energy),
        "soc_terminal": abs(f.soc_end.iloc[-1]-plant.terminal_energy) if plant.terminal_energy is not None else 0.,
        "soc_lower": max(0., plant.energy_min-f.soc_end.min()),
        "soc_upper": max(0., f.soc_end.max()-plant.energy_max),
        "charge_limit": max(0., f.charge.max()-plant.charge_max, -f.charge.min()),
        "discharge_limit": max(0., f.discharge.max()-plant.discharge_max, -f.discharge.min()),
        "mutual_exclusivity": np.minimum(f.charge, f.discharge).abs().max(),
        "pcc_import": max(0., f.grid.max()-plant.import_max),
        "pcc_export": max(0., -plant.export_max-f.grid.min()),
        "curtailment": max(0., (f.curtailment-f.pv).max(), -f.curtailment.min()),
        "critical_load": max(0., (f.unserved-plant.flexible_fraction*f.load).max(), -f.unserved.min()),
        "settlement_identity": np.abs(f.imbalance-f.grid+f.p+f.id_increment).max(),
        "cost_reconstruction": np.abs(cost-f.total_cost).max(),
        "da_limit": max(0., f.p.max()-plant.da_max, -f.p.min()),
        "intraday_limit": max(0., f.id_increment.abs().max()-plant.id_max),
    }
    return {"passed": all(value <= tolerance for value in violation.values()),
            "maximum_errors": {k: float(v) for k, v in violation.items()},
            "reconstructed_cost_eur": float(cost.sum()), "unserved_mwh": float(f.unserved.sum()),
            "curtailed_mwh": float(f.curtailment.sum()), "throughput_mwh": float((f.charge+f.discharge).sum())}


def summarize_daily(frame):
    success = frame[frame.status.eq("complete")]
    result = {"attempted_days": len(frame), "complete_days": len(success), "failed_days": len(frame)-len(success)}
    if len(success):
        c = success.cost_eur.to_numpy()
        result.update({"mean_eur": float(c.mean()), "cvar90_eur": empirical_cvar(c, .9),
                       "cvar95_eur": empirical_cvar(c, .95), "seconds_per_day": float(success.seconds.mean()),
                       "unserved_mwh": float(success.unserved_mwh.sum())})
    result["comparison_valid"] = result["failed_days"] == 0
    return result


def paired_block_bootstrap(left, right, block=10, replications=10000, seed=71, alpha=.95):
    """Paired date blocks; same sampled dates for both methods and all seeds.

    Columns represent algorithm seeds, not independent market replications.
    Report conditional uncertainty on the observed market series.
    """
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.ndim != 2 or len(a) < block:
        raise ValueError("paired [date,seed] arrays and sufficient days required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("failures/missing dates must be resolved before inference")
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty((replications, 3))
    for r in range(replications):
        starts = rng.integers(n, size=int(np.ceil(n/block)))
        sample = ((starts[:, None]+np.arange(block))%n).ravel()[:n]
        aa, bb = a[sample], b[sample]
        diffs[r, 0] = aa.mean()-bb.mean()
        for k, level in enumerate((.9, .95), 1):
            diffs[r, k] = np.mean([empirical_cvar(aa[:, s], level)-empirical_cvar(bb[:, s], level) for s in range(a.shape[1])])
    observed = [a.mean()-b.mean()]+[np.mean([empirical_cvar(a[:, s], level)-empirical_cvar(b[:, s], level)
                                           for s in range(a.shape[1])]) for level in (.9, .95)]
    result = {}
    for k, metric in enumerate(("mean", "cvar90", "cvar95")):
        lower, upper = np.quantile(diffs[:, k], [(1-alpha)/2, 1-(1-alpha)/2])
        # Centered bootstrap test of zero paired difference.
        p = (1+np.count_nonzero(np.abs(diffs[:, k]-observed[k]) >= abs(observed[k])))/(replications+1)
        result[metric] = {"difference": float(observed[k]), "ci": [float(lower), float(upper)], "p_value": float(p)}
    return {"block_days": block, "replications": replications, "seeds": a.shape[1], "metrics": result}


def holm_adjust(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]).clip(0, 1)
    out = np.empty_like(p)
    out[order] = adjusted
    return out
