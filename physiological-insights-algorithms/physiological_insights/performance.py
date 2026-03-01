"""Baseline detection, score trajectories, and trend analysis."""

import numpy as np
import pandas as pd
from scipy import stats


def _linear_slope(series: pd.Series) -> float | None:
    """Compute linear regression slope of a series indexed by ordinal position."""
    clean = series.dropna()
    if len(clean) < 2:
        return None
    x = np.arange(len(clean), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, clean.values)
    return float(slope)


def _daily_agg(df: pd.DataFrame, test_type: str) -> pd.DataFrame:
    """Aggregate a single test type to daily means."""
    sub = df[df["type"] == test_type].copy()
    if sub.empty:
        return pd.DataFrame()
    daily = sub.groupby("date").agg(
        mean=("score", "mean"),
        min=("score", "min"),
        max=("score", "max"),
        count=("score", "size"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date")


def analyse_performance(df: pd.DataFrame) -> dict:
    """Return performance analysis dict from test-results DataFrame."""
    result: dict = {}

    # --- Baseline ---
    baselines = df[df["is_baseline"] & (df["type"] == "READY")]
    if baselines.empty:
        baselines = df[df["is_baseline"]]
    if not baselines.empty:
        result["ready_baseline"] = float(baselines["score"].iloc[0])
    else:
        ready_scores = df.loc[df["type"] == "READY", "score"].dropna()
        result["ready_baseline"] = float(ready_scores.max()) if len(ready_scores) else None

    # --- Per-type stats ---
    for ttype in ("READY", "AGILITY", "FOCUS"):
        scores = df.loc[df["type"] == ttype, "score"].dropna()
        if scores.empty:
            result[ttype.lower()] = {"peak": None, "floor": None, "mean": None, "iqr": [None, None]}
            continue
        q1, q3 = float(scores.quantile(0.25)), float(scores.quantile(0.75))
        result[ttype.lower()] = {
            "peak": float(scores.max()),
            "floor": float(scores.min()),
            "mean": float(scores.mean()),
            "iqr": [q1, q3],
        }

    # --- Daily Ready trajectory ---
    daily_ready = _daily_agg(df, "READY")
    if not daily_ready.empty:
        daily_ready["rolling_7d"] = daily_ready["mean"].rolling(7, min_periods=1).mean()
        result["daily_ready"] = daily_ready.to_dict(orient="records")

        result["ready_7d_slope"] = _linear_slope(daily_ready["mean"].tail(7))
        result["ready_overall_slope"] = _linear_slope(daily_ready["mean"])

        baseline = result.get("ready_baseline")
        if baseline:
            daily_ready["pct_baseline"] = daily_ready["mean"] / baseline * 100
            result["daily_ready"] = daily_ready.to_dict(orient="records")
    else:
        result["daily_ready"] = []
        result["ready_7d_slope"] = None
        result["ready_overall_slope"] = None

    # --- Daily Agility / Focus ---
    for ttype in ("AGILITY", "FOCUS"):
        daily = _daily_agg(df, ttype)
        result[f"daily_{ttype.lower()}"] = daily.to_dict(orient="records") if not daily.empty else []

    # --- Self-report aggregation per day ---
    sr_cols = ["stress", "sleepiness", "sharpness"]
    existing_sr = [c for c in sr_cols if c in df.columns]
    if existing_sr:
        daily_sr = df.groupby("date")[existing_sr].mean().reset_index()
        daily_sr["date"] = pd.to_datetime(daily_sr["date"])
        result["daily_self_report"] = daily_sr.sort_values("date").to_dict(orient="records")
    else:
        result["daily_self_report"] = []

    # --- Weekly aggregation ---
    ready_df = df[df["type"] == "READY"].copy()
    if not ready_df.empty:
        ready_df["iso_week"] = ready_df["local_time"].dt.isocalendar().week.astype(int)
        ready_df["iso_year"] = ready_df["local_time"].dt.isocalendar().year.astype(int)
        ready_df["week_label"] = ready_df["iso_year"].astype(str) + "-W" + ready_df["iso_week"].astype(str).str.zfill(2)

        weekly = ready_df.groupby("week_label").agg(
            ready_mean=("score", "mean"),
            ready_min=("score", "min"),
            ready_max=("score", "max"),
            test_count=("score", "size"),
            date_min=("date", "min"),
            date_max=("date", "max"),
        ).reset_index().sort_values("week_label")

        baseline = result.get("ready_baseline")
        if baseline:
            weekly["ready_pct_baseline"] = weekly["ready_mean"] / baseline * 100

        # Week-over-week diffs
        weekly["ready_change_pct"] = weekly["ready_mean"].pct_change() * 100

        # Merge in self-report weekly means
        if existing_sr:
            all_tests = df.copy()
            all_tests["iso_week"] = all_tests["local_time"].dt.isocalendar().week.astype(int)
            all_tests["iso_year"] = all_tests["local_time"].dt.isocalendar().year.astype(int)
            all_tests["week_label"] = all_tests["iso_year"].astype(str) + "-W" + all_tests["iso_week"].astype(str).str.zfill(2)
            weekly_sr = all_tests.groupby("week_label")[existing_sr].mean().reset_index()
            weekly = weekly.merge(weekly_sr, on="week_label", how="left")

        # Agility weekly
        agility_df = df[df["type"] == "AGILITY"].copy()
        if not agility_df.empty:
            agility_df["iso_week"] = agility_df["local_time"].dt.isocalendar().week.astype(int)
            agility_df["iso_year"] = agility_df["local_time"].dt.isocalendar().year.astype(int)
            agility_df["week_label"] = agility_df["iso_year"].astype(str) + "-W" + agility_df["iso_week"].astype(str).str.zfill(2)
            weekly_ag = agility_df.groupby("week_label")["score"].mean().reset_index().rename(columns={"score": "agility_mean"})
            weekly = weekly.merge(weekly_ag, on="week_label", how="left")

        # Focus weekly
        focus_df = df[df["type"] == "FOCUS"].copy()
        if not focus_df.empty:
            focus_df["iso_week"] = focus_df["local_time"].dt.isocalendar().week.astype(int)
            focus_df["iso_year"] = focus_df["local_time"].dt.isocalendar().year.astype(int)
            focus_df["week_label"] = focus_df["iso_year"].astype(str) + "-W" + focus_df["iso_week"].astype(str).str.zfill(2)
            weekly_fc = focus_df.groupby("week_label")["score"].mean().reset_index().rename(columns={"score": "focus_mean"})
            weekly = weekly.merge(weekly_fc, on="week_label", how="left")

        result["weekly"] = weekly.to_dict(orient="records")
    else:
        result["weekly"] = []

    return result
