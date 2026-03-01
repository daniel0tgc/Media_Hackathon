"""Whoop-level sleep session analysis: debt tracking, architecture, recovery, performance score."""

import numpy as np
import pandas as pd

_DEEP_TARGET_PCT = (15, 20)
_REM_TARGET_PCT = (20, 25)
_LIGHT_TARGET_PCT = (50, 60)
_EFFICIENCY_GOOD = 85
_DEBT_REPAY_RATE = 4


def _classify_session_type(df: pd.DataFrame) -> pd.Series:
    """Tag each session as 'primary' or 'nap'.

    Rules:
    - If two sessions share the same night_date, the shorter is a nap.
    - Any session starting between 10:00-18:00 local time is a likely nap.
    - Sessions under 3 hours are naps unless they're the only session that day.
    """
    types = pd.Series("primary", index=df.index)

    if "sleep_start_local" in df.columns:
        start_hour = df["sleep_start_local"].dt.hour
        types[start_hour.between(10, 17)] = "nap"

    if "total_sleep_time_min" in df.columns:
        for date, group in df.groupby("night_date"):
            if len(group) > 1:
                longest_idx = group["total_sleep_time_min"].idxmax()
                for idx in group.index:
                    if idx != longest_idx:
                        types[idx] = "nap"

    return types


def analyse_sleep_sessions(df: pd.DataFrame) -> dict:
    """Full analysis of sleep sessions CSV."""
    if df is None or df.empty:
        return _empty_result()

    df = df.copy()
    df["session_type"] = _classify_session_type(df)

    all_nights = _build_night_summaries(df)
    primary_nights = [n for n in all_nights if n["session_type"] == "primary"]
    nap_nights = [n for n in all_nights if n["session_type"] == "nap"]

    primary_df = df[df["session_type"] == "primary"]

    debt = _analyse_sleep_debt(primary_df)
    recovery = _analyse_recovery(primary_df)
    performance = _compute_sleep_performance(primary_df, primary_nights)
    architecture = _aggregate_architecture(primary_nights, nap_nights)

    return {
        "nights": all_nights,
        "sleep_debt": debt,
        "recovery": recovery,
        "sleep_performance": performance,
        "architecture_summary": architecture,
    }


def _empty_result() -> dict:
    return {
        "nights": [],
        "sleep_debt": None,
        "recovery": None,
        "sleep_performance": None,
        "architecture_summary": None,
    }


def _build_night_summaries(df: pd.DataFrame) -> list[dict]:
    nights = []
    for _, row in df.iterrows():
        total = row.get("total_sleep_time_min") or 0
        deep = row.get("total_deep") or 0
        rem = row.get("total_rem") or 0
        light = row.get("total_light") or 0
        wake = row.get("total_wake") or 0
        session = row.get("total_session_time_min") or 0

        deep_pct = round(deep / total * 100, 1) if total > 0 else 0
        rem_pct = round(rem / total * 100, 1) if total > 0 else 0
        light_pct = round(light / total * 100, 1) if total > 0 else 0
        wake_pct = round(wake / session * 100, 1) if session > 0 else 0
        efficiency = round(total / session * 100, 1) if session > 0 else 0

        needed = row.get("sleep_needed_min")
        sufficiency = round(total / needed * 100, 1) if needed and needed > 0 else None

        nights.append({
            "night_date": str(row.get("night_date", "")),
            "session_type": row.get("session_type", "primary"),
            "sleep_start": str(row.get("sleep_start_local", "")),
            "sleep_end": str(row.get("sleep_end_local", "")),
            "total_sleep_min": round(total, 1),
            "session_time_min": round(session, 1),
            "deep_min": round(deep, 1),
            "deep_pct": deep_pct,
            "deep_on_target": _DEEP_TARGET_PCT[0] <= deep_pct <= _DEEP_TARGET_PCT[1],
            "rem_min": round(rem, 1),
            "rem_pct": rem_pct,
            "rem_on_target": _REM_TARGET_PCT[0] <= rem_pct <= _REM_TARGET_PCT[1],
            "light_min": round(light, 1),
            "light_pct": light_pct,
            "wake_min": round(wake, 1),
            "wake_pct": wake_pct,
            "efficiency_pct": efficiency,
            "sufficiency_pct": sufficiency,
            "sleep_needed_min": round(needed, 1) if needed else None,
            "sleep_debt_min": round(row["sleep_debt_min"], 1) if pd.notna(row.get("sleep_debt_min")) else None,
            "recovery_score": round(row["recovery_score"], 1) if pd.notna(row.get("recovery_score")) else None,
            "stress_score": round(row["stress_score"], 2) if pd.notna(row.get("stress_score")) else None,
            "avg_hrv_rmssd_ms": round(row["avg_hrv_rmssd_ms"], 1) if pd.notna(row.get("avg_hrv_rmssd_ms")) else None,
            "avg_hr_bpm": round(row["avg_hr_bpm_when_wake"], 1) if pd.notna(row.get("avg_hr_bpm_when_wake")) else None,
            "circadian_compliance": round(row["circadian_compliance"], 1) if pd.notna(row.get("circadian_compliance")) else None,
        })
    return nights


def _analyse_sleep_debt(df: pd.DataFrame) -> dict:
    if df.empty or "sleep_debt_min" not in df.columns:
        return {"current_debt_min": None, "current_debt_hours": None}

    debts = df["sleep_debt_min"].dropna()
    if debts.empty:
        return {"current_debt_min": None, "current_debt_hours": None}

    current = float(debts.iloc[-1])
    debt_hours = round(current / 60, 1)
    nights_to_repay = round(debt_hours * _DEBT_REPAY_RATE, 0) if debt_hours > 0 else 0

    needed = df["sleep_needed_min"].dropna()
    actual = df["total_sleep_time_min"].dropna()
    avg_deficit = round((needed.mean() - actual.mean()), 1) if len(needed) and len(actual) else None

    trend = "stable"
    if len(debts) >= 3:
        last_3 = debts.iloc[-3:]
        if last_3.iloc[-1] > last_3.iloc[0]:
            trend = "worsening"
        elif last_3.iloc[-1] < last_3.iloc[0]:
            trend = "improving"
    elif len(debts) >= 2:
        if debts.iloc[-1] > debts.iloc[0]:
            trend = "worsening"
        elif debts.iloc[-1] < debts.iloc[0]:
            trend = "improving"

    return {
        "current_debt_min": round(current, 1),
        "current_debt_hours": debt_hours,
        "nights_to_repay": int(nights_to_repay),
        "avg_nightly_deficit_min": avg_deficit,
        "trend": trend,
    }


def _analyse_recovery(df: pd.DataFrame) -> dict:
    if df.empty or "recovery_score" not in df.columns:
        return {"latest": None}

    scores = df["recovery_score"].dropna()
    if scores.empty:
        return {"latest": None}

    latest = float(scores.iloc[-1])
    mean_score = float(scores.mean())

    zone = "red" if latest < 34 else "yellow" if latest < 67 else "green"

    trend = "stable"
    if len(scores) >= 2:
        if scores.iloc[-1] > scores.iloc[-2]:
            trend = "improving"
        elif scores.iloc[-1] < scores.iloc[-2]:
            trend = "declining"

    green_count = (scores >= 67).sum()
    days_since_green = None
    for i in range(len(scores) - 1, -1, -1):
        if scores.iloc[i] >= 67:
            days_since_green = len(scores) - 1 - i
            break
    if days_since_green is None and len(scores) > 0:
        days_since_green = len(scores)

    return {
        "latest": round(latest, 1),
        "zone": zone,
        "personal_mean": round(mean_score, 1),
        "pct_of_mean": round(latest / mean_score * 100, 1) if mean_score > 0 else None,
        "trend": trend,
        "days_since_green_zone": days_since_green,
        "history": [round(float(s), 1) for s in scores],
    }


def _compute_sleep_performance(df: pd.DataFrame, nights: list[dict]) -> dict:
    if not nights:
        return None

    sufficiencies = [n["sufficiency_pct"] for n in nights if n["sufficiency_pct"] is not None]
    sufficiency_score = min(100, np.mean(sufficiencies)) if sufficiencies else None

    efficiencies = [n["efficiency_pct"] for n in nights]
    efficiency_score = min(100, np.mean(efficiencies)) if efficiencies else None

    bedtime_variance_min = None
    consistency_score = None
    if "sleep_start_local" in df.columns and len(df) >= 2:
        bedtime_hours = df["sleep_start_local"].dt.hour + df["sleep_start_local"].dt.minute / 60.0
        adjusted = bedtime_hours.apply(lambda h: h - 24 if h > 12 else h)
        bedtime_std = float(adjusted.std())
        bedtime_variance_min = round(bedtime_std * 60, 0)
        consistency_score = max(0, min(100, 100 - bedtime_std * 25))

    arch_scores = []
    for n in nights:
        deep_score = _target_score(n["deep_pct"], _DEEP_TARGET_PCT[0], _DEEP_TARGET_PCT[1])
        rem_score = _target_score(n["rem_pct"], _REM_TARGET_PCT[0], _REM_TARGET_PCT[1])
        arch_scores.append((deep_score + rem_score) / 2)
    architecture_score = np.mean(arch_scores) if arch_scores else None

    components = [s for s in [sufficiency_score, efficiency_score, consistency_score, architecture_score] if s is not None]
    composite = round(np.mean(components), 1) if components else None

    return {
        "composite_score": composite,
        "sufficiency": round(sufficiency_score, 1) if sufficiency_score is not None else None,
        "efficiency": round(efficiency_score, 1) if efficiency_score is not None else None,
        "consistency": round(consistency_score, 1) if consistency_score is not None else None,
        "architecture_quality": round(architecture_score, 1) if architecture_score is not None else None,
        "bedtime_variance_min": int(bedtime_variance_min) if bedtime_variance_min is not None else None,
    }


def _target_score(value: float, lo: float, hi: float) -> float:
    if lo <= value <= hi:
        return 100.0
    mid = (lo + hi) / 2
    distance = min(abs(value - lo), abs(value - hi))
    return max(0, 100 - distance * (100 / mid))


def _aggregate_architecture(primary_nights: list[dict], nap_nights: list[dict]) -> dict | None:
    if not primary_nights:
        return None

    avg_deep_pct = round(np.mean([n["deep_pct"] for n in primary_nights]), 1)
    avg_rem_pct = round(np.mean([n["rem_pct"] for n in primary_nights]), 1)

    rem_debt_min_3night = 0
    recent_primary = primary_nights[-3:]
    for n in recent_primary:
        target_rem_min = n["total_sleep_min"] * (_REM_TARGET_PCT[0] / 100)
        deficit = target_rem_min - n["rem_min"]
        if deficit > 0:
            rem_debt_min_3night += deficit
    rem_debt_min_3night = round(rem_debt_min_3night, 0)

    avg_bedtime = None
    avg_wake = None
    if primary_nights:
        bedtimes = [n["sleep_start"] for n in primary_nights if n["sleep_start"]]
        wakes = [n["sleep_end"] for n in primary_nights if n["sleep_end"]]
        if bedtimes:
            try:
                bt_times = pd.to_datetime(bedtimes)
                avg_h = bt_times.hour.astype(float) + bt_times.minute / 60.0
                adj = avg_h.map(lambda h: h - 24 if h > 12 else h)
                mean_h = adj.mean()
                if mean_h < 0:
                    mean_h += 24
                avg_bedtime = f"{int(mean_h):02d}:{int((mean_h % 1) * 60):02d}"
            except Exception:
                pass
        if wakes:
            try:
                wk_times = pd.to_datetime(wakes)
                avg_h = wk_times.hour.astype(float) + wk_times.minute / 60.0
                mean_h = avg_h.mean()
                avg_wake = f"{int(mean_h):02d}:{int((mean_h % 1) * 60):02d}"
            except Exception:
                pass

    return {
        "avg_deep_pct": avg_deep_pct,
        "avg_rem_pct": avg_rem_pct,
        "avg_light_pct": round(np.mean([n["light_pct"] for n in primary_nights]), 1),
        "avg_efficiency_pct": round(np.mean([n["efficiency_pct"] for n in primary_nights]), 1),
        "deep_target": f"{_DEEP_TARGET_PCT[0]}-{_DEEP_TARGET_PCT[1]}%",
        "rem_target": f"{_REM_TARGET_PCT[0]}-{_REM_TARGET_PCT[1]}%",
        "rem_debt_min_3night": int(rem_debt_min_3night),
        "avg_bedtime_local": avg_bedtime,
        "avg_wake_local": avg_wake,
        "session_type_breakdown": {
            "primary_nights": len(primary_nights),
            "nap_sessions": len(nap_nights),
        },
    }
