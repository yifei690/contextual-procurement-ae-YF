import numpy as np
import pytest
import torch
from covrisk.differentiable import BarrierSettings
from covrisk.distributions import construct_distribution
from covrisk.predictors import make_predictor
from covrisk.replay import replay_day, conditioned_remainder
from covrisk.tree import ScenarioTree


def make_pack(name):
    torch.manual_seed(153)
    model = make_predictor(name, history=24, horizon=4)
    inputs = torch.randn(5, 24, 5, dtype=torch.float64)
    future = torch.randn(5, 4, 6, dtype=torch.float64)
    mask = torch.ones(5, 4, dtype=torch.bool)
    query = model(inputs[:1], future[:1], mask[:1])
    historical = model(inputs[1:], future[1:], mask[1:])
    residuals = torch.randn(4, 4, 5, dtype=torch.float64)*.15
    center = torch.tensor([1., .3, 30., 25., 40.], dtype=torch.float64)
    scale = torch.tensor([.1, .1, 30., 30., 30.], dtype=torch.float64)
    pack = construct_distribution(query, historical, residuals, center, scale,
            torch.full((4,), 1.), .15, np.arange(4), ScenarioTree.regular(4, (2, 2)))
    actual = center.repeat(4, 1)
    actual[:, 0] += torch.tensor([.1, -.1, .05, -.05])
    actual[:, 4] = torch.tensor([-20., 100., 70., -10.])
    return model, pack, actual, scale


@pytest.mark.parametrize("architecture", ["bp", "lstm", "tcn", "transformer"])
def test_real_joint_gradient_without_forecast_score(architecture):
    torch.set_num_threads(1)
    model, pack, actual, scale = make_pack(architecture)
    result = replay_day(pack, actual, scale, smooth=True,
                        settings=BarrierSettings(mu=.001, tau=.0001, tolerance=1e-7, max_iterations=200))
    result.cost.backward()
    heads = {name: sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None)
             for name, head in (("location", model.heads.location), ("scale", model.heads.scale), ("embedding", model.heads.embedding))}
    assert min(heads.values()) > 1e-10
    encoder = [float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None]
    assert sum(encoder) > 1e-10
    assert all(record.get("adjoint_residual", 0) < 1e-6 for record in result.solver_records)


def test_replay_physical_and_no_future_price_input():
    _, pack, actual, scale = make_pack("bp")
    result = replay_day(pack, actual, scale)
    assert sum(r["total_cost"] for r in result.trajectory) == pytest.approx(float(result.cost)*1000, abs=1e-6)
    for row in result.trajectory:
        assert abs(row["grid"]+row["pv"]-row["curtailment"]+row["discharge"]+row["unserved"]-row["load"]-row["charge"]) < 1e-7
        assert row["charge"]*row["discharge"] < 1e-7
        assert row["curtailment"] <= row["pv"]+1e-7
    # Changing future final prices changes settlements but cannot change actions.
    altered = actual.clone()
    altered[:, 3:] += 1000
    changed = replay_day(pack, altered, scale)
    for left, right in zip(result.trajectory, changed.trajectory):
        for name in ("p", "id_increment", "charge", "discharge", "grid"):
            assert left[name] == pytest.approx(right[name], abs=1e-7)


def test_native_trees_cannot_be_mislabeled_as_jpo():
    with pytest.raises(ValueError, match="no native gradient JPO"):
        make_predictor("gbr", training="jpo")
