"""Physical load classification from IMU data and heart rate."""

import numpy as np
import pandas as pd


# Thresholds calibrated for wrist-worn 30-sec epoch data
_SEDENTARY_ENERGY = 50
_LIGHT_ENERGY = 200
_MODERATE_ENERGY = 600


def _epoch_energy(row: pd.Series) -> float:
    """Total accelerometer energy from energyPerSec columns."""
    cols = [c for c in row.index if "energyPerSec" in c and c.startswith("acc_")]
    if cols:
        vals = row[cols].dropna()
        return float(vals.sum()) if not vals.empty else 0.0
    rms_cols = [c for c in row.index if c.startswith("acc_") and "rms" in c.lower()]
    if rms_cols:
        vals = row[rms_cols].dropna()
        return float(np.sqrt((vals ** 2).sum())) if not vals.empty else 0.0
    return 0.0


def _classify_epoch(energy: float, hr: float) -> str:
    if energy < _SEDENTARY_ENERGY and hr < 80:
        return "sedentary"
    elif energy < _LIGHT_ENERGY:
        return "light"
    elif energy < _MODERATE_ENERGY or (80 <= hr < 100):
        return "moderate"
    else:
        return "vigorous"


def analyse_activity(df: pd.DataFrame) -> dict:
    """Classify epochs and compute daily physical load metrics."""
    wear = df[df["wear_mode"] == "wear_on"].copy()
    if wear.empty:
        return {"daily_activity": [], "exercise_sessions": []}

    wear["acc_energy"] = wear.apply(_epoch_energy, axis=1)
    hr_col = "heart_rate_mean" if "heart_rate_mean" in wear.columns else None
    hr_values = wear[hr_col].fillna(60) if hr_col else pd.Series(60, index=wear.index)

    wear["activity_level"] = [
        _classify_epoch(e, h) for e, h in zip(wear["acc_energy"], hr_values)
    ]

    wear["date"] = wear["datetime_et"].dt.date

    # Daily aggregation
    daily_list = []
    for date, day_df in wear.groupby("date"):
        counts = day_df["activity_level"].value_counts()
        total = len(day_df)
        moderate_vigorous = day_df[day_df["activity_level"].isin(["moderate", "vigorous"])]
        load_energy = float(moderate_vigorous["acc_energy"].sum()) if not moderate_vigorous.empty else 0

        # Steps and calories are cumulative counters — daily total = max − min
        steps = 0
        if "steps" in day_df.columns:
            s = day_df["steps"].dropna()
            if not s.empty:
                steps = max(0, float(s.max() - s.min()))

        calories = 0
        if "calories" in day_df.columns:
            c = day_df["calories"].dropna()
            if not c.empty:
                calories = max(0, float(c.max() - c.min()))

        daily_list.append({
            "date": str(date),
            "sedentary_epochs": int(counts.get("sedentary", 0)),
            "light_epochs": int(counts.get("light", 0)),
            "moderate_epochs": int(counts.get("moderate", 0)),
            "vigorous_epochs": int(counts.get("vigorous", 0)),
            "total_epochs": total,
            "moderate_vigorous_min": round((counts.get("moderate", 0) + counts.get("vigorous", 0)) * 0.5, 1),
            "physical_load": round(load_energy, 1),
            "exercise_load": _load_label(load_energy),
            "steps": steps,
            "calories": round(calories, 1),
        })

    # Exercise session detection: contiguous epochs with HR > 100 and high acc energy
    sessions = _detect_exercise_sessions(wear, hr_col)

    return {"daily_activity": daily_list, "exercise_sessions": sessions}


def _load_label(energy: float) -> str:
    if energy < 500:
        return "low"
    elif energy < 2000:
        return "moderate"
    else:
        return "high"


def _detect_exercise_sessions(df: pd.DataFrame, hr_col: str | None) -> list[dict]:
    """Find contiguous epochs of elevated HR + accelerometer energy."""
    if hr_col is None:
        return []

    mask = (df[hr_col] > 100) & (df["acc_energy"] > _MODERATE_ENERGY)
    exercise_epochs = df[mask]
    if exercise_epochs.empty:
        return []

    sessions = []
    idx_arr = exercise_epochs.index.values
    groups = np.split(idx_arr, np.where(np.diff(idx_arr) > 2)[0] + 1)

    for group in groups:
        if len(group) < 4:  # at least 2 min
            continue
        start = df.loc[group[0], "datetime_et"]
        end = df.loc[group[-1], "datetime_et"]
        dur = (end - start).total_seconds() / 60
        avg_hr = float(df.loc[group, hr_col].mean())
        sessions.append({
            "start": str(start),
            "end": str(end),
            "duration_min": round(dur, 1),
            "avg_hr": round(avg_hr, 1),
        })

    return sessions
