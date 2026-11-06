# Contextual Procurement Code and Case Data

This repository provides the core Python implementation for contextual, risk-aware electricity procurement with storage and a compact, single-day numerical case.

The example uses 24 hourly records from 2026-04-06 (base scenario) for H-DRO, X-PTO, and X-JPO. The archived trajectories and costs are derived from a completed evaluation. `examples/show_case.py` checks the hourly record counts and reconciles each hourly cost sum with the archived daily total. It does not retrain a predictor or rerun the optimizer.

## Run

Python 3.11 or newer is required.

```sh
python -m pip install -e ".[dev]"
python -B -X utf8 examples/show_case.py
python -B -X utf8 examples/solve_case.py
python -B -X utf8 -m pytest -q
```

The model implementation is in `src/covrisk`. The selected case data, frozen day-ahead scenarios, and configuration are in `examples/`. `solve_case.py` re-solves the three day-ahead optimization problems with SciPy/HiGHS and compares their objectives with the archived optima. It requires no commercial solver license. See [DATA_ACCESS.md](DATA_ACCESS.md) for data provenance and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the scope of the demonstration.

This code archive does not contain a manuscript, manuscript figures, reference lists, full-period result tables, raw market-data archives, or model checkpoints. Reproducing a new end-to-end experiment requires reacquiring the upstream data and supplying the corresponding experiment configuration and trained model artifacts.
