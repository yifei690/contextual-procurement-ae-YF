"""Publication-time embargoes and immutable residual-cache specifications."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def chronology_audit(data, config):
    spec = config["data"]
    stages = {}
    for name in ("joint_training", "calibration", "validation", "final_test"):
        start, end = spec[name+"_start"], spec[name+"_end"]
        if start > end:
            raise ValueError("reversed interval: "+name)
        ids = data.interval(start, end)
        if not len(ids):
            raise ValueError("empty interval: "+name)
        stages[name] = {"start": start, "end": end, "eligible_days": len(ids),
                        "requested_days": len(pd.date_range(start, end)),
                        "first_cutoff_ns": int(data.cutoff_ns[ids].min()),
                        "latest_label_available_ns": int(data.available_ns[ids].max())}
    fit_ids = data.interval(data.dates[0], spec["prediction_fit_end"])
    if not len(fit_ids):
        raise ValueError("empty prediction fitting interval")
    checks = []
    previous = "prediction_fit"
    available = int(data.available_ns[fit_ids].max())
    for name, row in stages.items():
        passed = available <= row["first_cutoff_ns"]
        checks.append({"from": previous, "to": name, "passed": passed,
                       "latest_label_available_utc": pd.Timestamp(available, tz="UTC").isoformat(),
                       "next_first_cutoff_utc": pd.Timestamp(row["first_cutoff_ns"], tz="UTC").isoformat(),
                       "margin_hours": (row["first_cutoff_ns"]-available)/3.6e12})
        previous, available = name, row["latest_label_available_ns"]
    return {"passed": all(row["passed"] for row in checks), "stages": stages, "checks": checks,
            "scope": "delivery intervals plus declared label-release lag; no vintage reconstruction"}


def require_chronology(data, config):
    report = chronology_audit(data, config)
    if not report["passed"]:
        bad = [row["from"]+" -> "+row["to"] for row in report["checks"] if not row["passed"]]
        raise ValueError("unreleased labels at next stage: "+", ".join(bad))
    return report


def residual_spec(name, seed, data, start, end, refit_days, epochs):
    # Hash every source that determines the fitted forecasts or residuals.
    source = hashlib.sha256()
    for name_ in ("training.py", "predictors.py", "distributions.py", "data.py", "protocol.py"):
        source.update((Path(__file__).parent/name_).read_bytes())
    return {"version": 2, "architecture": name, "seed": int(seed),
            "dataset_sha256": data.metadata["dataset_sha256"], "start": start, "end": end,
            "refit_days": int(refit_days), "fit_epochs": int(epochs),
            "source_sha256": source.hexdigest()}


def load_residual_cache(path, expected, data):
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        if metadata.get("cache_spec") != expected or str(z["dataset_hash"]) != expected["dataset_sha256"]:
            raise ValueError("residual cache specification mismatch; choose a fresh cache path")
        ids, residuals = z["ids"].copy(), z["residuals"].copy()
    if ids.dtype.kind not in "iu" or ids.ndim != 1 or len(ids) == 0:
        raise ValueError("invalid residual cache ids")
    if np.any(ids < 0) or np.any(ids >= len(data.dates)) or np.any(np.diff(ids) <= 0):
        raise ValueError("invalid residual cache ids")
    if residuals.shape != tuple(data.target[ids].shape) or not np.isfinite(residuals).all():
        raise ValueError("invalid residual cache values")
    if any(not expected["start"] <= data.dates[int(i)] <= expected["end"] for i in ids):
        raise ValueError("residual cache dates outside requested interval")
    return ids, residuals, metadata
