"""Regression checks for label-release embargoes and residual cache provenance."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from covrisk.protocol import chronology_audit, require_chronology, residual_spec, load_residual_cache


def fake_data():
    dates = pd.date_range("2023-03-15", "2026-09-30")
    labels = [d.date().isoformat() for d in dates]
    data = SimpleNamespace(dates=labels,
        cutoff_ns=np.array([(d-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize("Europe/Berlin").value for d in dates]),
        available_ns=np.array([((d+pd.Timedelta(days=1)).tz_localize("Europe/Berlin")+pd.Timedelta(days=60)).value for d in dates]),
        target=torch.zeros(len(dates), 25, 5, dtype=torch.float64),
        metadata={"dataset_sha256": "fixture"})
    data.interval = lambda start,end: np.array([i for i,d in enumerate(labels) if start <= d <= end], dtype=int)
    return data


def configuration():
    return {"data": {"prediction_fit_end": "2023-10-31",
        "joint_training_start": "2024-01-01", "joint_training_end": "2024-06-30",
        "calibration_start": "2024-09-01", "calibration_end": "2024-11-30",
        "validation_start": "2025-02-01", "validation_end": "2025-10-31",
        "final_test_start": "2026-01-01", "final_test_end": "2026-09-30"}}


def test_release_time_embargo_rejects_calendar_disjoint_leakage():
    data, config = fake_data(), configuration()
    assert require_chronology(data, config)["passed"]
    config["data"]["validation_end"] = "2025-12-31"
    assert not chronology_audit(data, config)["passed"]
    with pytest.raises(ValueError, match="validation -> final_test"):
        require_chronology(data, config)
    config = configuration()
    config["data"]["calibration_start"] = "2024-07-01"
    with pytest.raises(ValueError, match="joint_training -> calibration"):
        require_chronology(data, config)


def test_cache_rejects_wrong_model_budget_dates_and_source(tmp_path):
    data = fake_data()
    spec = residual_spec("bp", 17, data, "2023-07-15", "2023-10-31", 90, 10)
    ids = data.interval(spec["start"], spec["end"])
    path = tmp_path/"residuals.npz"
    np.savez_compressed(path, ids=ids, residuals=np.zeros((len(ids),25,5)),
                        dataset_hash="fixture", metadata=json.dumps({"cache_spec": spec}))
    got, _, _ = load_residual_cache(path, spec, data)
    np.testing.assert_array_equal(got, ids)
    for key, value in (("architecture","lstm"),("seed",43),("fit_epochs",30),
                       ("end","2024-06-30"),("refit_days",30),("source_sha256","changed")):
        with pytest.raises(ValueError, match="specification mismatch"):
            load_residual_cache(path, {**spec, key:value}, data)
    np.savez_compressed(path, ids=ids, residuals=np.zeros((len(ids),25,5)),
                        dataset_hash="fixture", metadata=json.dumps({"architecture":"bp"}))
    with pytest.raises(ValueError, match="specification mismatch"):
        load_residual_cache(path, spec, data)
