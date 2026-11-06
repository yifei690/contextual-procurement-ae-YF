# Demonstration Scope

`examples/show_case.py` reads the archived 24-hour trajectories for H-DRO, X-PTO, and X-JPO on 2026-04-06. It verifies that each method has one record for every hour and that the sum of hourly costs matches the stored daily cost to numerical tolerance. It then prints the daily costs and the X-PTO/X-JPO differences relative to H-DRO.

`examples/solve_case.py` reconstructs the three day-ahead mixed-integer DRO problems from the frozen forecast scenarios, weights, distances, tree nodes, and ambiguity radii. It solves each problem using SciPy's HiGHS backend, checks feasibility and optimal status, and compares the objective with the original optimum to EUR 0.01. These forecast-scenario objective values differ from realized daily settlement costs. Alternative optimal actions and numerical differences across solvers are possible.

Neither script retrains predictors or replays all intraday decisions. The Python source in `src/covrisk` is provided for inspection and reuse. New end-to-end runs require source data, an experiment configuration, and trained artifacts that are not part of this compact archive.

Run the included core tests with:

```sh
python -B -X utf8 -m pytest -q
```
