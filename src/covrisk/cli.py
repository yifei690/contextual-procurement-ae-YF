"""Configuration entry point; commands preserve provenance and failure states."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import copy
import numpy as np
import torch
from .data import DayBatch, load_germany, prepare_days
from .differentiable import BarrierSettings
from .physical import Plant
from .risk import RiskSpec
from .replay import replay_day
from .training import new_model, fit_forecast, crossfit_residuals, DistributionFactory, finetune, model_hash

ROOT = Path(__file__).resolve().parents[2]


def prepare(args):
    frame, audit = load_germany(ROOT/"data/raw/de")
    data = prepare_days(frame, end=args.end)
    output = ROOT/"data/processed"
    output.mkdir(parents=True, exist_ok=True)
    data.save(output/"germany.npz")
    (output/"audit.json").write_text(json.dumps({"sources": audit, **data.metadata}, indent=2), encoding="utf-8")
    print(json.dumps({"days": len(data.dates), "first": data.dates[0], "last": data.dates[-1],
                      "excluded": len(data.metadata["exclusions"]), "sources": audit}, indent=2))


def pilot(args):
    torch.set_num_threads(1)
    data = DayBatch.load(ROOT/"data/processed/germany.npz")
    output = ROOT/"runs"/f"pilot_{args.model}_{args.seed}_{data.metadata['dataset_sha256'][:8]}"
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = new_model(args.model, args.seed, data)
    train = data.interval("2023-03-15", "2023-12-31")
    # Pilot only: preprocessing and training predates the 2024 diagnostics.
    log = fit_forecast(model, data, train, args.epochs, args.seed)
    torch.save(model.state_dict(), output/"pretrained.pt")
    residual_ids, residuals, folds = crossfit_residuals(args.model, args.seed, data,
        end="2023-12-31", epochs=args.epochs, path=output/"residuals.npz")
    frozen = copy.deepcopy(model).eval()
    factory = DistributionFactory(data, residual_ids, residuals, frozen, leaves=args.leaves, seed=args.seed)
    candidates = data.interval("2024-03-01", "2024-06-30")
    if not len(candidates):
        raise ValueError("pilot dates unavailable")
    day = int(candidates[0])
    h = int(data.mask[day].sum())
    pack = factory(model, day)
    truth = data.observed[day, :h]
    settings = BarrierSettings(mu=args.mu, tau=args.tau, tolerance=1e-7, max_iterations=250)
    smooth = replay_day(pack, truth, data.scale, smooth=True, settings=settings)
    smooth.cost.backward()
    gradients = {key: float(sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None))
                 for key, module in (("location", model.heads.location), ("scale", model.heads.scale),
                                     ("embedding", model.heads.embedding), ("encoder", model.encoder))}
    exact = replay_day(factory(model, day), truth, data.scale, backend=args.backend)
    record = {"status": "pilot_complete_not_formal_results", "architecture": args.model,
              "seed": args.seed, "date": data.dates[day], "training_days": len(train),
              "residuals": len(residual_ids), "leaves": args.leaves, "gradient_l1": gradients,
              "smooth_cost_eur": float(smooth.cost.detach())*1000, "exact_cost_eur": float(exact.cost.detach())*1000,
              "smooth_seconds": smooth.seconds, "exact_seconds": exact.seconds,
              "layer_diagnostics": smooth.solver_records, "exact_diagnostics": exact.solver_records,
              "elapsed_seconds": time.perf_counter()-start, "checkpoint_sha256": model_hash(model),
              "data_sha256": data.metadata["dataset_sha256"], "pretraining": log, "folds": folds}
    (output/"pilot.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(exact.trajectory).to_csv(output/"exact_trajectory.csv", index=False)
    pd.DataFrame(smooth.trajectory).to_csv(output/"smooth_trajectory.csv", index=False)
    if args.joint_step:
        finetune(model, factory, candidates[:1], "jpo", 1, args.seed, output/"joint_step", settings=settings,
                 batch_size=1, max_batches=1)
    print(json.dumps({k: v for k, v in record.items() if k not in ("layer_diagnostics", "exact_diagnostics", "pretraining", "folds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="covrisk")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare-data")
    p.add_argument("--end", default="2026-09-30")
    p.set_defaults(function=prepare)
    p = commands.add_parser("pilot")
    p.add_argument("--model", choices=["bp", "lstm", "tcn", "transformer"], default="bp")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--leaves", type=int, default=3)
    p.add_argument("--mu", type=float, default=.0001)
    p.add_argument("--tau", type=float, default=.0001)
    p.add_argument("--backend", choices=["highs", "gurobi"], default="gurobi")
    p.add_argument("--joint-step", action="store_true")
    p.set_defaults(function=pilot)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
