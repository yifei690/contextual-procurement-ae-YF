from dataclasses import replace
import numpy as np
import pytest
import torch
from covrisk.canonical import Builder
from covrisk.differentiable import BarrierSettings, solve_differentiable
from covrisk.physical import Plant, build_procurement, scenario_costs
from covrisk.risk import RiskSpec, adversary, dual_value, empirical_cvar
from covrisk.solvers import solve_exact, solve_cuts
from covrisk.tree import ScenarioTree


def fixture_model(theta=None, plant=Plant(), formulation="dual"):
    tree = ScenarioTree.regular(4, (2, 2))
    rng = np.random.default_rng(41)
    values = rng.normal(size=(4, 4, 5))
    values[..., 0] = 1+0.1*values[..., 0]
    values[..., 1] = 0.2+0.03*values[..., 1]
    values[..., 2:] = 20+40*values[..., 2:]
    base = torch.tensor(values, dtype=torch.float64)
    if theta is None:
        theta = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    shift = torch.stack([theta[0]*0.1, theta[0]*0.01, theta[1]*10, theta[1]*20, theta[1]*5])
    paths = tree.project(base*(1+0.05*theta[0])+shift)
    weights = torch.softmax(torch.arange(4, dtype=torch.float64)*theta[2], 0)
    scaled = paths/torch.tensor([1, 1, 100, 100, 100], dtype=torch.float64)
    distance = (scaled[:, None]-scaled[None, :]).square().mean((2, 3))
    risk = RiskSpec(0.3, (0.8, 0.9), (0.5, 0.5))
    model = build_procurement(paths, tree, weights, distance, 0.2, plant=plant, risk=risk, formulation=formulation)
    return model, weights, distance, risk


def test_transport_duality_negative_cost_and_monotonicity():
    rng = np.random.default_rng(987)
    for _ in range(10):
        points = rng.normal(size=(4, 2))
        d = ((points[:, None]-points[None, :])**2).sum(-1)
        w = rng.dirichlet(np.ones(4))
        c = rng.normal(size=4)*100
        for rho in (0, 0.3, 0.8):
            risk = RiskSpec(rho)
            values = []
            for radius in (0, 0.2, 3.):
                p = adversary(c, w, d, radius, risk)
                q = dual_value(c, w, d, radius, risk)
                assert abs(p["value"]-q) < 1e-7
                assert abs(p["omega"].sum()-1) < 1e-8
                values.append(q)
            assert np.min(np.diff(values)) > -1e-7


def test_zero_radius_nominal_risk():
    c = np.array([-50, -1, 3, 9.])
    w = np.array([.4, .3, .2, .1])
    d = (np.arange(4)[:, None]-np.arange(4)[None, :])**2
    risk = RiskSpec()
    expected = .7*(c@w)+.3*sum(b*empirical_cvar(c, a, w) for a,b in zip(risk.alpha,risk.beta))
    assert adversary(c, w, d, 0, risk)["value"] == pytest.approx(expected)


def test_exact_dual_and_cut_master_agree():
    direct, w, d, risk = fixture_model()
    master, *_ = fixture_model(formulation="cuts")
    exact = solve_exact(direct)
    cuts, history = solve_cuts(master, w.numpy(), d.detach().numpy(), .2, risk)
    assert exact.status == cuts.status == "optimal"
    assert exact.objective == pytest.approx(cuts.objective, abs=2e-6)
    checked = adversary(scenario_costs(direct, exact.x), w.numpy(), d.numpy(), .2, risk)
    assert checked["value"] == pytest.approx(exact.objective, abs=1e-7)
    assert all(row["lower"] <= row["upper"]+1e-6 for row in history)
    refs = direct.meta["refs"]
    assert np.max(exact.x[refs["c"]]*exact.x[refs["d"]]) < 1e-7
    assert max(exact.residuals.values()) < 1e-6


def test_nonanticipative_actions_and_prefix_rejection():
    model, *_ = fixture_model()
    refs = model.meta["refs"]
    assert len(np.unique(refs["c"][:, 0])) == 1
    assert len(np.unique(refs["g"][:, 0])) == 2
    tree = ScenarioTree.regular(4, (2, 2))
    with pytest.raises(ValueError, match="unshared prefix"):
        tree.check_paths(np.arange(4*4*5).reshape(4, 4, 5))


def test_no_storage():
    plant = replace(Plant(), charge_max=0, discharge_max=0)
    model, *_ = fixture_model(plant=plant)
    sol = solve_exact(model)
    assert sol.status == "optimal"
    assert np.max(np.abs(sol.x[model.meta["refs"]["c"]])) < 1e-8
    assert np.max(np.abs(sol.x[model.meta["refs"]["d"]])) < 1e-8


def test_sparse_adjoint_all_coefficient_types():
    theta = torch.tensor([.2, -.1, .1, .3, .1], dtype=torch.float64, requires_grad=True)
    def f(par):
        b = Builder()
        x = b.var("x", -2, 2, objective=par[0])
        y = b.var("y", -2, 2, objective=.2)
        b.le({x: 1+par[1], y: .2}, 1.1+par[2])
        b.eq({x: 1, y: 1+par[3]}, .3+par[4])
        solution, _ = solve_differentiable(b.finish(), BarrierSettings(mu=.1, tau=.05, tolerance=1e-10))
        return solution
    assert torch.autograd.gradcheck(f, (theta,), eps=1e-5, atol=2e-5, rtol=2e-4)


def test_full_procurement_gradient():
    theta = torch.tensor([.1, .2, -.1], dtype=torch.float64, requires_grad=True)
    settings = BarrierSettings(mu=.01, tau=.002, tolerance=2e-8, max_iterations=200)
    def f(par):
        model, *_ = fixture_model(par)
        solution, info = solve_differentiable(model, settings)
        return solution[model.meta["p"]]
    assert torch.autograd.gradcheck(f, (theta,), eps=2e-5, atol=5e-4, rtol=3e-3)


def test_reused_kkt_matches_eliminated_system_and_preserves_factors():
    from scipy import sparse
    from covrisk.differentiable import _KKTSystem

    rng = np.random.default_rng(312)
    a = rng.normal(size=(7, 4))
    mu, tau = .03, .02
    # Both constrained and unconstrained adjoints are used by the layer.
    for equality_count in (0, 2):
        e = rng.normal(size=(equality_count, 4))
        system = _KKTSystem(sparse.csr_matrix(a), sparse.csr_matrix(e), mu, tau)
        saved = []
        for slack in (np.linspace(.1, 1., 7), np.linspace(.8, .2, 7)):
            hessian = mu*np.eye(4)+a.T@np.diag(tau/slack**2)@a
            dense = np.block([[hessian, e.T],
                              [e, np.zeros((equality_count, equality_count))]])
            rhs = rng.normal(size=4+equality_count)
            expected = np.linalg.solve(dense, rhs)
            factor = system.factor(slack)
            np.testing.assert_allclose(factor.solve(rhs), expected, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(factor.solve(rhs, trans="T"), expected, rtol=1e-10, atol=1e-10)
            saved.append((factor, rhs, expected))
        # A later Newton step must not invalidate an earlier saved factor.
        for factor, rhs, expected in saved:
            np.testing.assert_allclose(factor.solve(rhs), expected, rtol=1e-10, atol=1e-10)
