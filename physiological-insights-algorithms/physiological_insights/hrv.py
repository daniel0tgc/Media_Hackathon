"""HRV analysis: personal baseline, diurnal profile, rolling trends."""

import numpy as np
import pandas as pd
from scipy import stats


_CONFIDENCE_THRESHOLD = 0.7


def _valid_hrv(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to epochs with reliable HRV readings."""
    mask = (
        (df["wear_mode"] == "wear_on")
        & (df["cardio_RMSSD_ms"] > 0)
        & (df["cardio_confidence_median"] > _CONFIDENCE_THRESHOLD)
    )
    return df.loc[mask].copy()


def analyse_hrv(df: pd.DataFrame) -> dict:
    """Compute HRV metrics from decoded-metrics DataFrame."""
    result: dict = {}
    valid = _valid_hrv(df)

    if valid.empty:
        return {
            "rmssd_baseline": None,
            "rmssd_sleep_peak": None,
            "morning_rmssd": None,
            "rmssd_pct_baseline": None,
            "sdnn_mean": None,
            "diurnal_profile": [],
            "rmssd_7d_slope": None,
            "hr_range": None,
        }

    rmssd = valid["cardio_RMSSD_ms"]

    # Personal RMSSD baseline: mean of top 20%
    top20_threshold = rmssd.quantile(0.80)
    result["rmssd_baseline"] = float(rmssd[rmssd >= top20_threshold].mean())

    # Sleep RMSSD peak: max 30-min rolling mean where HR < 65
    sleep_mask = valid["heart_rate_mean"] < 65
    sleep_valid = valid.loc[sleep_mask]
    if not sleep_valid.empty:
        rolling_rmssd = sleep_valid["cardio_RMSSD_ms"].rolling(60, min_periods=10).mean()
        result["rmssd_sleep_peak"] = float(rolling_rmssd.max()) if not rolling_rmssd.isna().all() else None
    else:
        result["rmssd_sleep_peak"] = None

    # Morning RMSSD: mean in last 60 min of recording where wear_on
    # (proxy — actual morning detection uses sleep offset from sleep module)
    last_ts = valid["datetime"].max()
    morning_window = valid[valid["datetime"] >= last_ts - pd.Timedelta(hours=1)]
    result["morning_rmssd"] = float(morning_window["cardio_RMSSD_ms"].mean()) if not morning_window.empty else None

    if result["morning_rmssd"] and result["rmssd_baseline"]:
        result["rmssd_pct_baseline"] = round(result["morning_rmssd"] / result["rmssd_baseline"] * 100, 1)
    else:
        result["rmssd_pct_baseline"] = None

    # SDNN
    sdnn = valid["cardio_SDNN_ms"]
    result["sdnn_mean"] = float(sdnn.mean()) if not sdnn.isna().all() else None

    # Diurnal profile: hourly mean RMSSD
    valid_copy = valid.copy()
    valid_copy["hour"] = valid_copy["datetime_et"].dt.hour
    hourly = valid_copy.groupby("hour")["cardio_RMSSD_ms"].mean().reset_index()
    hourly.columns = ["hour", "rmssd_mean"]
    result["diurnal_profile"] = hourly.to_dict(orient="records")

    # HR range
    hr = valid["heart_rate_mean"]
    hr_valid = hr[hr > 0]
    if not hr_valid.empty:
        result["hr_range"] = {"min": float(hr_valid.min()), "max": float(hr_valid.max()), "mean": float(hr_valid.mean())}
    else:
        result["hr_range"] = None

    # RMSSD daily means for trend (group by date)
    valid_copy["date"] = valid_copy["datetime_et"].dt.date
    daily_rmssd = valid_copy.groupby("date")["cardio_RMSSD_ms"].mean()
    if len(daily_rmssd) >= 2:
        x = np.arange(len(daily_rmssd), dtype=float)
        slope, _, _, _, _ = stats.linregress(x, daily_rmssd.values)
        result["rmssd_7d_slope"] = float(slope)
    else:
        result["rmssd_7d_slope"] = None

    # Full time series for visualization (downsample to 5-min for efficiency)
    ts = valid[["datetime_et", "cardio_RMSSD_ms", "heart_rate_mean"]].copy()
    ts = ts.set_index("datetime_et").resample("5min").mean().dropna(how="all").reset_index()
    result["timeseries"] = ts.to_dict(orient="records")

    return result
