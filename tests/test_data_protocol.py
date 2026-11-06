import numpy as np
import pandas as pd
from covrisk.data import prepare_days


def test_dst_days_and_availability_perturbation():
    # Synthetic inputs exercise timestamp logic only; they are not case data.
    index = pd.date_range("2022-12-01", "2023-11-02", freq="h", tz="UTC")
    rng = np.random.default_rng(71)
    frame = pd.DataFrame(rng.uniform(.2, 1., (len(index), 5)), index=index,
                         columns=["load", "pv", "day_ahead", "intraday", "imbalance"])
    data = prepare_days(frame, start="2023-03-25", end="2023-10-30", normalization_end="2023-02-01")
    spring = data.dates.index("2023-03-26")
    autumn = data.dates.index("2023-10-29")
    assert int(data.mask[spring].sum()) == 23
    assert int(data.mask[autumn].sum()) == 25
    cutoff = pd.Timestamp(data.cutoff_ns[autumn], tz="UTC")
    altered = frame.copy()
    altered.loc[cutoff:, :] += 500
    check = prepare_days(altered, start="2023-03-25", end="2023-10-30", normalization_end="2023-02-01")
    assert np.array_equal(data.history[autumn], check.history[autumn])
    assert np.array_equal(data.future[autumn], check.future[autumn])
    assert not np.array_equal(data.observed[autumn], check.observed[autumn])
