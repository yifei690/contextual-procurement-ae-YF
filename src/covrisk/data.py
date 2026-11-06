"""Public-data parsing, explicit availability lags, and DST-aware market days.

Archived revised data are not historical publication vintages. Availability
lags below are conservative benchmark assumptions, recorded in every manifest.
They must not be described as a verified real-time vintage reconstruction.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

FIELDS = ("load", "pv", "day_ahead", "intraday", "imbalance")
DEFAULT_LAGS = (48, 48, 48, 48, 24*60)  # hours since interval start


def _tso_frame(path, kind):
    frame = pd.read_csv(path, sep=";", na_values=["N.A.", "n.a.", "-", ""])
    index = pd.to_datetime(frame["Datum"]+" "+frame["von"], format="%d.%m.%Y %H:%M", utc=True)
    if kind == "rebap":
        names = [c for c in frame if c.startswith("reBAP")]
        vals = [pd.to_numeric(frame[c].astype(str).str.replace(",", ".", regex=False), errors="coerce") for c in names]
        if len(vals) != 2 or not np.allclose(vals[0], vals[1], equal_nan=True):
            raise ValueError("asymmetric imbalance prices require a dual-price adapter")
        value = vals[0]
    else:
        value = pd.to_numeric(frame.iloc[:, -1], errors="coerce")
    return pd.Series(value.to_numpy(), index=index, name=kind)


def load_germany(raw_root):
    root = Path(raw_root)
    all_series = {k: [] for k in FIELDS}
    for path in sorted(root.glob("power_????.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        index = pd.to_datetime(obj["unix_seconds"], unit="s", utc=True)
        for kind, name in (("load", "Load"), ("pv", "Solar")):
            series = next(s for s in obj["production_types"] if s["name"] == name)
            all_series[kind].append(pd.Series(series["data"], index=index, dtype=float))
    for path in sorted(root.glob("da_????.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        all_series["day_ahead"].append(pd.Series(obj["price"], index=pd.to_datetime(obj["unix_seconds"], unit="s", utc=True), dtype=float))
    for path in sorted(root.glob("rebap_*.csv")):
        all_series["imbalance"].append(_tso_frame(path, "rebap"))
    for path in sorted((root/"ip_daily").glob("????-??-??.csv")):
        all_series["intraday"].append(_tso_frame(path, "ipindex"))
    result, audit = {}, {}
    for kind, parts in all_series.items():
        if not parts:
            raise ValueError(f"missing public source: {kind}")
        series = pd.concat(parts).sort_index()
        duplicates = int(series.index.duplicated().sum())
        # Overlapping annual endpoints must agree; never pick an inconsistent revision silently.
        repeated = series[series.index.duplicated(keep=False)]
        if len(repeated) and (repeated.groupby(level=0).max()-repeated.groupby(level=0).min()).max() > 1e-8:
            raise ValueError(f"inconsistent overlapping revisions for {kind}")
        series = series[~series.index.duplicated(keep="first")]
        count = series.resample("h").count()
        hourly = series.resample("h").mean()
        # DA changes from hourly to quarter-hour products: accept exactly 1 or 4,
        # all other sources require four realized intervals per modeled hour.
        expected = count.isin([1, 4]) if kind == "day_ahead" else count.eq(4)
        hourly.loc[~expected] = np.nan
        result[kind] = hourly
        audit[kind] = {"source_rows": len(series), "overlap_rows": duplicates,
                       "valid_hours": int(hourly.notna().sum()), "missing_hours": int(hourly.isna().sum()),
                       "first_utc": str(series.index[0]), "last_utc": str(series.index[-1])}
    frame = pd.DataFrame(result).sort_index()
    return frame, audit


@dataclass
class DayBatch:
    dates: list
    history: torch.Tensor
    future: torch.Tensor
    target: torch.Tensor
    observed: torch.Tensor
    mask: torch.Tensor
    pv_upper: torch.Tensor
    cutoff_ns: np.ndarray
    available_ns: np.ndarray
    delivery_ns: np.ndarray
    center: torch.Tensor
    scale: torch.Tensor
    metadata: dict

    def select(self, ids):
        ids = np.atleast_1d(ids)
        return self.history[ids], self.future[ids], self.mask[ids]

    def interval(self, start, end):
        return np.asarray([i for i, d in enumerate(self.dates) if start <= d <= end], int)

    def save(self, path):
        arrays = {k: getattr(self, k).numpy() for k in ("history", "future", "target", "observed", "mask", "pv_upper", "center", "scale")}
        np.savez_compressed(path, **arrays, dates=np.array(self.dates), cutoff_ns=self.cutoff_ns,
                            available_ns=self.available_ns, delivery_ns=self.delivery_ns,
                            metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            tensors = {k: torch.from_numpy(z[k]) for k in ("history", "future", "target", "observed", "mask", "pv_upper", "center", "scale")}
            return cls([str(value) for value in z["dates"]], **tensors, cutoff_ns=z["cutoff_ns"], available_ns=z["available_ns"],
                       delivery_ns=z["delivery_ns"], metadata=json.loads(str(z["metadata"])))


def prepare_days(frame, normalization_end="2023-03-31", start="2023-03-15", end="2026-09-30",
                 history_hours=168, lags=DEFAULT_LAGS):
    raw = frame[list(FIELDS)].copy()
    fit = raw.loc[:normalization_end]
    if len(fit) < 1000:
        raise ValueError("insufficient normalization history")
    # Public national shapes -> a clearly labeled simulated site, no contest constants.
    load_divisor = float(fit["load"].quantile(.99)/1.5)
    pv_divisor = float(fit["pv"].max()/1.2)
    raw["load"] /= load_divisor
    raw["pv"] /= pv_divisor
    center = raw.loc[:normalization_end].mean().to_numpy()
    scale = raw.loc[:normalization_end].std().clip(lower=1e-3).to_numpy()
    normalized = (raw-center)/scale
    dates, histories, futures, targets, observed, masks, uppers, cutoffs, available, deliveries = ([] for _ in range(10))
    exclusions = []
    for day in pd.date_range(start, end):
        label = day.date().isoformat()
        begin = day.tz_localize("Europe/Berlin")
        finish = (day+pd.Timedelta(days=1)).tz_localize("Europe/Berlin")
        index = pd.date_range(begin, finish, freq="h", inclusive="left").tz_convert("UTC")
        # Submission at 11:00 local time on D-1, before day-ahead market closure.
        cutoff = (day-pd.Timedelta(days=1)+pd.Timedelta(hours=11)).tz_localize("Europe/Berlin").tz_convert("UTC")
        h = len(index)
        truth = raw.reindex(index).to_numpy()
        if not np.isfinite(truth).all():
            exclusions.append({"date": label, "reason": "incomplete_target", "missing_fields": [FIELDS[i] for i in np.where(~np.isfinite(truth).all(0))[0]]})
            continue
        history = np.empty((history_hours, len(FIELDS)))
        valid_history = True
        latest = []
        for f, lag in enumerate(lags):
            last = cutoff-pd.Timedelta(hours=lag)-pd.Timedelta(hours=1)
            times = pd.date_range(end=last, periods=history_hours, freq="h")
            vals = normalized.iloc[:, f].reindex(times).to_numpy()
            if not np.isfinite(vals).all():
                valid_history = False
                break
            history[:, f] = vals
            latest.append(int((times[-1]+pd.Timedelta(hours=lag)).value))
        if not valid_history:
            exclusions.append({"date": label, "reason": "incomplete_available_history"})
            continue
        local = index.tz_convert("Europe/Berlin")
        hours = local.hour.to_numpy()
        doy = local.dayofyear.to_numpy()
        weekday = local.dayofweek.to_numpy()
        calendar = np.stack([np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
                             np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                             np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)], 1)
        future = np.zeros((25, 6))
        future[:h] = calendar
        target, obs = np.zeros((25, len(FIELDS))), np.zeros((25, len(FIELDS)))
        target[:h], obs[:h] = (truth-center)/scale, truth
        mask = np.arange(25) < h
        # Conservative night mask from solar geometry (latitude 51 degrees,
        # longitude 10 degrees); cap is fixed from normalization period.
        utc_hour = index.hour.to_numpy()+.5
        solar_hour = utc_hour+10/15
        declination = np.deg2rad(23.44)*np.sin(2*np.pi*(284+doy)/365.25)
        elevation = np.arcsin(np.sin(np.deg2rad(51))*np.sin(declination)+
                              np.cos(np.deg2rad(51))*np.cos(declination)*np.cos(np.pi*(solar_hour-12)/12))
        pv_upper = np.zeros(25)
        pv_upper[:h] = np.where(elevation > np.deg2rad(-8), 1.6, 0.)
        times = np.zeros(25, np.int64)
        times[:h] = index.asi8
        dates.append(label)
        histories.append(history)
        futures.append(future)
        targets.append(target)
        observed.append(obs)
        masks.append(mask)
        uppers.append(pv_upper)
        cutoffs.append(cutoff.value)
        # Entire residual trajectory becomes available only after slowest field.
        available.append((index[-1]+pd.Timedelta(hours=max(lags)+1)).value)
        deliveries.append(times)
        if max(latest) > cutoff.value:
            raise AssertionError("future covariate")
    if not dates:
        raise ValueError("no complete causal days; inspect source missingness")
    metadata = {"market": "DE-LU", "timezone": "Europe/Berlin", "resolution": "hourly averaged benchmark",
                "normalization_end": normalization_end, "load_divisor": load_divisor, "pv_divisor": pv_divisor,
                "history_hours": history_hours, "availability_lag_hours": list(lags),
                "information_cutoff": "D-1 11:00 Europe/Berlin", "exclusions": exclusions,
                "data_vintage": "final archive with declared lags, not publication-vintage data",
                "settlement": "simulated hourly contract indexed to public ex-post price series",
                "requested_start": start, "requested_end": end,
                "eligible_days": len(dates), "unavailable_days": len(exclusions),
                "dataset_sha256": hashlib.sha256((raw.to_csv()+normalization_end+str(lags)+str(history_hours)).encode()).hexdigest()}
    tensor = lambda values: torch.tensor(np.asarray(values), dtype=torch.float64)
    return DayBatch(dates, tensor(histories), tensor(futures), tensor(targets), tensor(observed),
                    torch.tensor(np.asarray(masks), dtype=torch.bool), tensor(uppers), np.asarray(cutoffs),
                    np.asarray(available), np.asarray(deliveries), tensor(center), tensor(scale), metadata)
