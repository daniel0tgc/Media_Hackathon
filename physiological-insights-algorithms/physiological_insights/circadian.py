"""Circadian performance curve via cosinor model fit on Ready scores."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _cosinor(t: np.ndarray, mesor: float, amplitude: float, acrophase: float) -> np.ndarray:
    """24-hour cosine model: y = MESOR + amplitude * cos(2*pi*(t - acrophase)/24)"""
    return mesor + amplitude * np.cos(2 * np.pi * (t - acrophase) / 24)


def analyse_circadian(df: pd.DataFrame, sleep_df=None) -> dict:
    """Fit a cosinor model to Ready scores by time-of-day."""
    ready = df[df["type"] == "READY"].copy()
    if ready.empty or "hour" not in ready.columns:
        return {
            "cosinor_fit": False,
            "mesor": None,
            "amplitude": None,
            "acrophase_hour": None,
            "estimated_peak_window": None,
            "worst_window": None,
            "hourly_bins": [],
            "cosinor_n": 0,
            "peak_score_variance": None,
            "chronotype_estimate": None,
        }

    hours = ready["hour"].values
    scores = ready["score"].values

    # 2-hour bin aggregation
    bins = list(range(0, 24, 2))
    bin_labels = [f"{b:02d}:00-{b + 2:02d}:00" for b in bins]
    ready["hour_bin"] = pd.cut(ready["hour"], bins=bins + [24], labels=bin_labels, right=False)
    hourly = ready.groupby("hour_bin", observed=True)["score"].agg(["mean", "count"]).reset_index()
    hourly.columns = ["window", "mean_score", "test_count"]

    result: dict = {
        "hourly_bins": hourly.to_dict(orient="records"),
    }

    # Cosinor fit
    valid = ~np.isnan(scores)
    t_fit = hours[valid]
    y_fit = scores[valid]

    result["cosinor_n"] = int(len(t_fit))

    if len(t_fit) < 6:
        result["cosinor_fit"] = False
        result["mesor"] = float(np.mean(y_fit)) if len(y_fit) else None
        result["amplitude"] = None
        result["acrophase_hour"] = None
        result["estimated_peak_window"] = None
        result["worst_window"] = None
        result["peak_score_variance"] = None
        result["chronotype_estimate"] = None
        _add_sleep_timing(result, sleep_df)
        return result

    mesor_guess = float(np.mean(y_fit))
    amp_guess = float((np.max(y_fit) - np.min(y_fit)) / 2)
    acro_guess = float(t_fit[np.argmax(y_fit)])

    try:
        popt, _ = curve_fit(
            _cosinor, t_fit, y_fit,
            p0=[mesor_guess, amp_guess, acro_guess],
            maxfev=10000,
        )
        mesor, amplitude, acrophase = popt
        amplitude = abs(amplitude)
        acrophase = acrophase % 24

        # Peak window: range where fitted curve > 90% of peak value
        t_dense = np.linspace(0, 24, 240)
        y_dense = _cosinor(t_dense, mesor, amplitude, acrophase)
        peak_val = y_dense.max()
        threshold = mesor + 0.9 * amplitude  # 90% of way from mean to peak
        above = t_dense[y_dense >= threshold]
        if len(above) >= 2:
            peak_start = float(above[0])
            peak_end = float(above[-1])
            result["estimated_peak_window"] = f"{int(peak_start):02d}:00-{int(peak_end):02d}:00"
        else:
            result["estimated_peak_window"] = f"{int(acrophase):02d}:00-{int(acrophase + 2) % 24:02d}:00"

        # Worst window: where curve is lowest
        worst_hour = float(t_dense[np.argmin(y_dense)])
        result["worst_window"] = f"{int(worst_hour):02d}:00-{int(worst_hour + 2) % 24:02d}:00"

        result["cosinor_fit"] = True
        result["mesor"] = round(float(mesor), 2)
        result["amplitude"] = round(float(amplitude), 2)
        result["acrophase_hour"] = round(float(acrophase), 2)

        # Store fitted curve for visualization
        result["fitted_curve"] = [
            {"hour": float(t_dense[i]), "score": float(y_dense[i])}
            for i in range(len(t_dense))
        ]

        # Peak score variance: std of scores within the estimated peak window
        if len(above) >= 2:
            peak_mask = ready["hour"].between(float(above[0]), float(above[-1]))
            peak_scores = ready.loc[peak_mask, "score"]
            result["peak_score_variance"] = round(float(peak_scores.var()), 1) if len(peak_scores) >= 2 else None
        else:
            result["peak_score_variance"] = None

        # Chronotype from acrophase
        result["chronotype_estimate"] = _chronotype_from_acrophase(acrophase)

    except (RuntimeError, ValueError):
        result["cosinor_fit"] = False
        result["mesor"] = mesor_guess
        result["amplitude"] = None
        result["acrophase_hour"] = None
        result["estimated_peak_window"] = None
        result["worst_window"] = None
        result["peak_score_variance"] = None
        result["chronotype_estimate"] = None

    _add_sleep_timing(result, sleep_df)
    return result


def _chronotype_from_acrophase(acrophase_hour: float) -> str:
    if acrophase_hour < 11:
        return "morning_lark"
    elif acrophase_hour < 15:
        return "intermediate"
    else:
        return "evening_owl"


def _add_sleep_timing(result: dict, sleep_df) -> None:
    """Add natural sleep onset/wake and bedtime variance from sleep sessions."""
    if sleep_df is None or sleep_df.empty:
        result["natural_sleep_onset_local"] = None
        result["natural_wake_local"] = None
        result["bedtime_variance_min"] = None
        return

    import pandas as _pd
    df = sleep_df.copy()

    # Classify naps and filter to primary sessions for timing calculations
    if "sleep_start_local" in df.columns:
        start_hour = df["sleep_start_local"].dt.hour
        is_nap = start_hour.between(10, 17)
        df = df[~is_nap]

    if df.empty:
        result["natural_sleep_onset_local"] = None
        result["natural_wake_local"] = None
        result["bedtime_variance_min"] = None
        return

    if "sleep_start_local" in df.columns:
        bedtime_hours = df["sleep_start_local"].dt.hour + df["sleep_start_local"].dt.minute / 60.0
        adjusted = bedtime_hours.apply(lambda h: h - 24 if h > 12 else h)
        mean_bt = float(adjusted.mean())
        if mean_bt < 0:
            mean_bt += 24
        result["natural_sleep_onset_local"] = f"{int(mean_bt):02d}:{int((mean_bt % 1) * 60):02d}"
        result["bedtime_variance_min"] = round(float(adjusted.std()) * 60, 0) if len(adjusted) >= 2 else None
    else:
        result["natural_sleep_onset_local"] = None
        result["bedtime_variance_min"] = None

    if "sleep_end_local" in df.columns:
        wake_hours = df["sleep_end_local"].dt.hour + df["sleep_end_local"].dt.minute / 60.0
        mean_wk = float(wake_hours.mean())
        result["natural_wake_local"] = f"{int(mean_wk):02d}:{int((mean_wk % 1) * 60):02d}"
    else:
        result["natural_wake_local"] = None
