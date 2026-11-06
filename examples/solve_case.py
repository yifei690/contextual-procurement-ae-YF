"""Re-solve the frozen day-ahead problems from one seed-43 numerical case."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covrisk.physical import Plant, build_procurement
from covrisk.risk import RiskSpec
from covrisk.solvers import solve_exact
from covrisk.tree import ScenarioTree


def main():
    torch.set_num_threads(1)
    folder = Path(__file__).resolve().parent
    config = json.loads((folder / "case_config.json").read_text(encoding="utf-8"))
    risk = RiskSpec(**config["risk"])
    print("Frozen day-ahead optimization | 2026-04-06 | seed 43 | SciPy/HiGHS")
    with np.load(folder / "case_2026-04-06_day_ahead.npz", allow_pickle=False) as data:
        for method, settings in config["methods"].items():
            tree = ScenarioTree(data[method + "_pre"], data[method + "_post"])
            tree.validate()
            tensor = lambda key: torch.as_tensor(data[method + "_" + key], dtype=torch.float64)
            program = build_procurement(
                tensor("projected_paths"), tree, tensor("weights"), tensor("distance"),
                float(data[method + "_radius"]), plant=Plant(**settings["plant"]),
                risk=risk, initial_energy=float(data[method + "_state"]))
            result = solve_exact(program, backend="highs", time_limit=30)
            if result.status != "optimal":
                raise RuntimeError(f"{method}: solver status {result.status}")
            recorded = settings["recorded_day_ahead_objective_keur"]
            difference = abs(result.objective - recorded) * 1000
            if difference > 0.01:
                raise RuntimeError(f"{method}: objective differs from archived optimum by EUR {difference}")
            print(f"{method:6} | objective EUR {result.objective * 1000:10.4f} "
                  f"| difference EUR {difference:.6g} | residual {max(result.residuals.values()):.2g}")
    print("These are forecast-scenario objectives; realized daily costs are shown by show_case.py.")


if __name__ == "__main__":
    main()
