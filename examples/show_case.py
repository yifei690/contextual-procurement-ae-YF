"""Validate and summarize the packaged one-day case data."""

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAJECTORY = ROOT / "seed43_2026-04-06.csv"
SUMMARY = ROOT / "seed43_2026-04-06_summary.csv"
METHODS = ("H-DRO", "X-PTO", "X-JPO")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    trajectories = defaultdict(list)
    for row in read_csv(TRAJECTORY):
        trajectories[row["method"]].append(row)

    summaries = {row["method"]: row for row in read_csv(SUMMARY)}
    totals = {}
    for method in METHODS:
        rows = sorted(trajectories[method], key=lambda row: int(row["hour_index"]))
        hours = [int(row["hour_index"]) for row in rows]
        if hours != list(range(24)):
            raise ValueError(f"{method}: expected one record for each hour 0-23")
        if method not in summaries:
            raise ValueError(f"{method}: missing daily summary")

        total = sum(float(row["hourly_cost_eur"]) for row in rows)
        archived = float(summaries[method]["total_cost_eur"])
        if abs(total - archived) > 1e-6:
            raise ValueError(f"{method}: hourly costs do not reconcile with daily total")
        if summaries[method]["physical_audit_passed"].lower() != "true":
            raise ValueError(f"{method}: archived physical audit did not pass")
        totals[method] = total

    baseline = totals["H-DRO"]
    print("Date: 2026-04-06 | seed: 43 | scenario: base | 24 hourly records")
    print(f"{'Method':<8} {'Cost (EUR)':>14} {'Difference vs H-DRO (EUR)':>28}")
    for method in METHODS:
        print(f"{method:<8} {totals[method]:>14.2f} {totals[method] - baseline:>28.2f}")


if __name__ == "__main__":
    main()
