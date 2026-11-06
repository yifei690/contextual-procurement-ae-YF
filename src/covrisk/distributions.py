"""Frozen residual trajectories, learned similarity, fixed trees, and W2 costs."""
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from .tree import ScenarioTree


@dataclass
class ParameterPack:
    paths: torch.Tensor
    weights: torch.Tensor
    distance: torch.Tensor
    radius: float
    tree: ScenarioTree
    effective_samples: torch.Tensor
    source_ids: np.ndarray


@dataclass(frozen=True)
class RadiusCalibrator:
    intercept: float = .02
    uncertainty_slope: float = .05
    concentration_slope: float = .05
    minimum: float = .01
    maximum: float = .25

    def __call__(self, frozen_scale, frozen_weights):
        scale = np.asarray(frozen_scale, float)
        weights = np.asarray(frozen_weights, float)
        effective = 1/np.square(weights).sum()
        return float(np.clip(self.intercept+self.uncertainty_slope*np.mean(scale)+
                             self.concentration_slope/np.sqrt(effective), self.minimum, self.maximum))


def physical_map(raw, pv_upper, delta=.01):
    load = delta*F.softplus(raw[..., 0]/delta)
    pv = delta*F.softplus(raw[..., 1]/delta)-delta*F.softplus((raw[..., 1]-pv_upper)/delta)
    return torch.cat([load[..., None], pv[..., None], raw[..., 2:]], -1)


def construct_distribution(forecast, historical_forecast, residuals, center, scale, pv_upper,
                           radius, source_ids, tree, temperature=1., probability_floor=.05,
                           uniform=False):
    """Inputs use one query and N historical examples, all evaluated by same net.

    Residuals and membership are frozen. Historical embeddings remain on the
    computation graph. The radius is an independently frozen float. Tree
    projection precedes transportation distances, preserving shared prefixes.
    """
    valid = forecast.mask[0].bool()
    location, spread = forecast.location[0, valid], forecast.scale[0, valid]
    if residuals.shape != (tree.pre.shape[0], len(location), location.shape[-1]):
        raise ValueError("frozen residual bank/tree mismatch")
    if not 0 < probability_floor <= 1 or temperature <= 0:
        raise ValueError("invalid conditional weight settings")
    raw = (location[None]+spread[None]*residuals)*scale+center
    paths = tree.project(physical_map(raw, pv_upper))
    distance_to_history = (historical_forecast.embedding-forecast.embedding).square().mean(-1)
    weights = torch.softmax(-distance_to_history/temperature, 0)
    weights = weights*(1-probability_floor)+probability_floor/len(weights)
    if uniform:
        weights = torch.ones_like(weights)/len(weights)
    standardized = paths/scale
    # Mean squared standardized distance: M=diag(1/scale)/sqrt(H*F), full rank.
    distance = (standardized[:, None]-standardized[None, :]).square().mean((2, 3))
    return ParameterPack(paths, weights, distance, float(radius), tree, 1/weights.square().sum(), np.asarray(source_ids))


def energy_score(paths, weights, truth, scale, smooth=0.):
    delta = (paths-truth)/scale
    first = torch.sqrt(delta.square().mean((1, 2))+smooth**2)
    pairs = (paths[:, None]-paths[None, :])/scale
    second = torch.sqrt(pairs.square().mean((2, 3))+smooth**2)
    return (weights*first).sum()-.5*(weights[:, None]*weights[None, :]*second).sum()


def forecast_loss(forecast, target):
    mask = forecast.mask[..., None].to(target.dtype)
    score = .5*((target-forecast.location)/forecast.scale).square()+torch.log(forecast.scale)
    return (score*mask).sum()/(mask.sum()*target.shape[-1])
