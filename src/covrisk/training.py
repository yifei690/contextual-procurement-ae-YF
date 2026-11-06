"""Chronological residual construction and budget-matched PTO/JPO training."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .data import DayBatch
from .differentiable import BarrierSettings
from .distributions import RadiusCalibrator, construct_distribution, energy_score, forecast_loss
from .physical import Plant
from .predictors import make_predictor
from .protocol import residual_spec, load_residual_cache
from .replay import replay_day
from .risk import RiskSpec
from .tree import ScenarioTree


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def fit_forecast(model, data, ids, epochs, seed, learning_rate=.001, batch_size=32):
    if len(ids) < 10:
        raise ValueError("at least ten complete days are required for fitting")
    if not isinstance(model, torch.nn.Module):
        model.fit(data.history[ids].numpy(), data.future[ids].numpy(), data.target[ids].numpy(), data.mask[ids].numpy())
        # Initial positive scale: a common adaptation, not a native tree distribution.
        out = model(*data.select(ids))
        errors = data.target[ids]-out.location
        model.scale = np.maximum(.05, torch.sqrt((errors.square()*data.mask[ids, :, None]).sum((0, 1))/data.mask[ids].sum()).numpy())
        return []
    model.train()
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    history = []
    for epoch in range(epochs):
        sums, count = 0., 0
        for batch in np.array_split(rng.permutation(ids), max(1, int(np.ceil(len(ids)/batch_size)))):
            optimizer.zero_grad()
            output = model(*data.select(batch))
            loss = forecast_loss(output, data.target[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite forecasting loss")
            loss.backward()
            optimizer.step()
            sums += float(loss.detach())*len(batch)
            count += len(batch)
        history.append({"epoch": epoch, "forecast_loss": sums/count})
    model.eval()
    return history


def new_model(name, seed, data):
    torch.manual_seed(seed)
    if name in ("ar", "gbr"):
        return make_predictor(name, horizon=data.mask.shape[1], seed=seed)
    return make_predictor(name, history=data.history.shape[1], horizon=data.mask.shape[1])


def crossfit_residuals(name, seed, data, end="2024-06-30", refit_days=90, epochs=30, path=None):
    """Forward expanding-window forecasts; every fit label was available at cutoff.

    Preprocessing fitted through March 2023 is available before residual forecasts
    begin in July 2023, including the 60-day imbalance lag. Incomplete/unavailable
    days remain explicitly recorded.
    """
    start = "2023-07-15"
    spec = residual_spec(name, seed, data, start, end, refit_days, epochs)
    if path is not None and Path(path).exists():
        ids, values, metadata = load_residual_cache(path, spec, data)
        return ids, torch.from_numpy(values), metadata
    candidates = data.interval(start, end)
    blocks = []
    remaining = candidates.tolist()
    while remaining:
        first = remaining[0]
        stop = (pd.Timestamp(data.dates[first])+pd.Timedelta(days=refit_days)).date().isoformat()
        block = [i for i in remaining if data.dates[i] < stop]
        blocks.append(block)
        remaining = [i for i in remaining if i not in set(block)]
    residual_ids, residuals, records = [], [], []
    for block in blocks:
        cutoff = data.cutoff_ns[block[0]]
        train = np.flatnonzero(data.available_ns <= cutoff)
        if len(train) < 30:
            records.append({"start": data.dates[block[0]], "status": "insufficient_historical_labels", "training_days": len(train)})
            continue
        model = new_model(name, seed, data)
        fit_forecast(model, data, train, epochs, seed)
        with torch.no_grad():
            prediction = model(*data.select(block))
            error = (data.target[block]-prediction.location)/prediction.scale
        residual_ids.extend(block)
        residuals.append(error)
        records.append({"start": data.dates[block[0]], "end": data.dates[block[-1]],
                        "training_days": len(train), "latest_training_available_ns": int(data.available_ns[train].max()),
                        "cutoff_ns": int(cutoff), "status": "complete"})
    if not residuals:
        raise ValueError("no causally valid residual folds")
    ids = np.asarray(residual_ids, int)
    values = torch.cat(residuals)
    metadata = {"architecture": name, "seed": seed, "folds": records, "fit_epochs": epochs,
                "refit_days": refit_days, "frozen": True, "cache_spec": spec}
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=ids, residuals=values.numpy(), metadata=json.dumps(metadata),
                            dataset_hash=data.metadata["dataset_sha256"])
    return ids, values, metadata


class DistributionFactory:
    def __init__(self, data, residual_ids, residuals, pretrained, leaves=9, seed=17, calibrator=RadiusCalibrator()):
        self.data, self.residual_ids, self.residuals = data, residual_ids, residuals
        self.pretrained = pretrained
        self.leaves, self.seed, self.calibrator = leaves, seed, calibrator
        self.membership, self.radius_cache = {}, {}

    def selected(self, day):
        if day not in self.membership:
            eligible = np.flatnonzero(self.data.available_ns[self.residual_ids] <= self.data.cutoff_ns[day])
            if len(eligible) < self.leaves:
                raise ValueError(f"{self.data.dates[day]}: fewer than {self.leaves} released residual trajectories")
            # Date-keyed common random numbers and fixed membership, no learned hard selection.
            rng = np.random.default_rng(self.seed+int(self.data.dates[day].replace("-", "")))
            choice = np.sort(rng.choice(eligible, self.leaves, replace=False))
            self.membership[day] = choice
        return self.membership[day]

    def __call__(self, model, day, uniform=False, fixed_radius=None):
        positions = self.selected(day)
        ids = self.residual_ids[positions]
        h = int(self.data.mask[day].sum())
        aligned = []
        for pos, idx in zip(positions, ids):
            length = int(self.data.mask[idx].sum())
            residual = self.residuals[pos, :length]
            # DST adaptation preserves each multivariate trajectory together.
            if length != h:
                residual = F.interpolate(residual.T[None], size=h, mode="linear", align_corners=True)[0].T
            aligned.append(residual)
        residuals = torch.stack(aligned)
        query = model(*self.data.select([day]))
        historical = model(*self.data.select(ids))
        branches = (3, self.leaves//3) if self.leaves%3 == 0 and self.leaves > 3 else (self.leaves,)
        tree = ScenarioTree.regular(h, branches)
        if day not in self.radius_cache:
            with torch.no_grad():
                baseline = self.pretrained(*self.data.select([day]))
                old = self.pretrained(*self.data.select(ids))
                weights = torch.softmax(-(old.embedding-baseline.embedding).square().mean(-1), 0)
                weights = .95*weights+.05/len(weights)
                self.radius_cache[day] = self.calibrator(baseline.scale[0, :h].numpy(), weights.numpy())
        radius = self.radius_cache[day] if fixed_radius is None else fixed_radius
        return construct_distribution(query, historical, residuals, self.data.center, self.data.scale,
                self.data.pv_upper[day, :h], radius, ids, tree, uniform=uniform)


def finetune(model, factory, ids, mode, epochs, seed, output, risk=RiskSpec(), plant=Plant(),
             lr=.0005, batch_size=4, settings=BarrierSettings(), lambda_es=.1,
             lambda_weight=.001, lambda_theta=1e-6, max_batches=None, initial_eta=None):
    """Matched SGD updates; JPO cost comes from causal realized replay, never self-valued risk.

    A failed day aborts this training run and is persisted, rather than becoming
    a zero gradient or a selectively excluded sample. Numerical remedies must be
    applied in development and recorded before final-test freezing.
    """
    if mode not in ("pto", "jpo", "heads", "stop_gradient"):
        raise ValueError("unknown training mode")
    if not model.supports_jpo:
        raise ValueError("traditional adapter cannot enter gradient training")
    model.configure_training("heads" if mode == "heads" else "full")
    eta_value = torch.full((len(risk.alpha),), .5, dtype=torch.float64) if initial_eta is None else torch.as_tensor(initial_eta, dtype=torch.float64).clone()
    if eta_value.shape != (len(risk.alpha),) or not torch.isfinite(eta_value).all():
        raise ValueError("invalid initial risk thresholds")
    eta = torch.nn.Parameter(eta_value)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad]+[eta], lr=lr)
    rng = np.random.default_rng(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = []
    data = factory.data
    manifest = {"mode": mode, "seed": seed, "epochs": epochs, "batch_size": batch_size,
                "learning_rate": lr, "max_batches": max_batches,
                "data_sha256": data.metadata["dataset_sha256"], "initial_model_sha256": model_hash(model),
                "initial_eta": eta_value.detach().tolist(),
                "settings": asdict(settings), "risk": asdict(risk), "plant": asdict(plant),
                "trainable_parameters": [name for name,p in model.named_parameters() if p.requires_grad],
                "training_dates": [data.dates[int(i)] for i in ids]}
    if (output/"training.jsonl").exists():
        raise ValueError("training directory already contains updates; use a fresh output")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for epoch in range(epochs):
        batches = np.array_split(rng.permutation(ids), max(1, int(np.ceil(len(ids)/batch_size))))
        if max_batches is not None:
            batches = batches[:max_batches]
        for step, batch in enumerate(batches):
            optimizer.zero_grad()
            begun = time.perf_counter()
            scores, realized, concentrations = [], [], []
            try:
                for day in batch:
                    pack = factory(model, int(day))
                    h = int(data.mask[day].sum())
                    truth = data.observed[day, :h]
                    scores.append(energy_score(pack.paths, pack.weights, truth, data.scale, smooth=1e-4))
                    concentrations.append((pack.weights*torch.log(pack.weights*len(pack.weights))).sum())
                    if mode != "pto":
                        replay = replay_day(pack, truth, data.scale, plant, risk, smooth=True, settings=settings,
                                            stop_optimization_gradient=(mode == "stop_gradient"))
                        realized.append(replay.cost)
                score = torch.stack(scores).mean()
                reg = sum(p.square().sum() for p in model.parameters())
                cost = torch.tensor(0., dtype=torch.float64)
                if realized:
                    costs = torch.stack(realized)
                    tails = torch.stack([eta[k]+(.01*F.softplus((costs-eta[k])/.01)).mean()/(1-alpha)
                                         for k, alpha in enumerate(risk.alpha)])
                    cost = (1-risk.rho)*costs.mean()+risk.rho*(torch.tensor(risk.beta)*tails).sum()
                # Forecast-score coefficient matched across PTO and JPO.
                loss = cost+lambda_es*score+lambda_weight*torch.stack(concentrations).mean()+lambda_theta*reg
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite joint objective")
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError("missing/nonfinite parameter gradient")
                grad_norm = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
                optimizer.step()
            except Exception as exc:
                failure = {"status": "failed", "epoch": epoch, "step": step,
                           "dates": [data.dates[int(i)] for i in batch], "error": repr(exc)}
                (output/"failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
                raise
            record = {"epoch": epoch, "step": step, "dates": [data.dates[int(i)] for i in batch],
                      "eta": eta.detach().tolist(), "loss": float(loss.detach()), "cost_loss": float(cost.detach()),
                      "es": float(score.detach()), "gradient_norm": grad_norm, "seconds": time.perf_counter()-begun}
            log.append(record)
            with (output/"training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record)+"\n")
        torch.save({"model": model.state_dict(), "eta": eta.detach(), "epoch": epoch,
                    "optimizer": optimizer.state_dict(), "hash": model_hash(model)}, output/f"epoch_{epoch:03d}.pt")
        print(json.dumps({"training": mode, "epoch": epoch, "last": log[-1]}), flush=True)
    return log
