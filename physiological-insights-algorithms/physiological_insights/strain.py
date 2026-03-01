"""Daily strain score from decoded metrics (Whoop-inspired, Borg 0-21 scale)."""

import math
import numpy as np
import pandas as pd

_ZONE_BOUNDS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
_ZONE_WEIGHTS = [0.0, 1.0, 2.0, 4.0, 8.0]

_STRAIN_CAP = 21.0
# Calibrated so: sedentary day ~2-5, moderate workout ~8-12, intense training 14-18+
_CV_DIVISOR = 80
_CV_SCALE = 5.5
_MUSC_DIVISOR = 5000
_MUSC_SCALE = 4.0


def _estimate_max_hr(df: pd.DataFrame) -> float:
    hr = df["heart_rate_mean"]
    valid = hr[hr > 0].dropna()
    if valid.empty:
        return 190.0
    return max(float(valid.quantile(0.99)), 150.0)


def _classify_zone(hr: float, max_hr: float) -> int:
    pct = hr / max_hr if max_hr > 0 else 0
    for i, bound in enumerate(_ZONE_BOUNDS[:-1]):
        if pct < bound:
            return i
    return 5


def _raw_to_borg(raw: float, divisor: float, scale: float) -> float:
    if raw <= 0:
        return 0.0
    score = scale * math.log(1 + raw / divisor)
    return min(_STRAIN_CAP, round(score, 1))


def analyse_strain(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "heart_rate_mean" not in df.columns:
        return {"daily_strain": [], "max_hr_est": None}

    wear = df[df["wear_mode"] == "wear_on"].copy() if "wear_mode" in df.columns else df.copy()
    if wear.empty:
        return {"daily_strain": [], "max_hr_est": None}

    max_hr = _estimate_max_hr(wear)
    wear["hr_zone"] = wear["heart_rate_mean"].apply(lambda h: _classify_zone(h, max_hr))
    wear["date"] = wear["datetime_et"].dt.date

    energy_cols = [c for c in wear.columns if "energyPerSec" in c and c.startswith("acc_")]
    if energy_cols:
        wear["acc_energy"] = wear[energy_cols].sum(axis=1)
    else:
        wear["acc_energy"] = 0.0

    daily_list = []
    for date, day_df in wear.groupby("date"):
        zone_counts = day_df["hr_zone"].value_counts()
        raw_cv = sum(
            zone_counts.get(z, 0) * 0.5 * _ZONE_WEIGHTS[z - 1]
            for z in range(1, 6)
        )
        cv_strain = _raw_to_borg(raw_cv, _CV_DIVISOR, _CV_SCALE)

        moderate_mask = day_df["hr_zone"] >= 3
        muscular_energy = float(day_df.loc[moderate_mask, "acc_energy"].sum()) if moderate_mask.any() else 0
        muscular_strain = _raw_to_borg(muscular_energy, _MUSC_DIVISOR, _MUSC_SCALE)

        combined = min(_STRAIN_CAP, round(cv_strain * 0.7 + muscular_strain * 0.3, 1))

        zone_minutes = {}
        for z in range(1, 6):
            zone_minutes[f"zone_{z}_min"] = round(zone_counts.get(z, 0) * 0.5, 1)

        sleep_adjustment_min = 0
        if combined >= 18:
            sleep_adjustment_min = 45
        elif combined >= 14:
            sleep_adjustment_min = 30
        elif combined >= 10:
            sleep_adjustment_min = 15

        daily_list.append({
            "date": str(date),
            "strain_score": combined,
            "cardiovascular_strain": cv_strain,
            "muscular_strain": muscular_strain,
            "strain_level": _strain_label(combined),
            **zone_minutes,
            "sleep_need_adjustment_min": sleep_adjustment_min,
        })

    return {
        "daily_strain": daily_list,
        "max_hr_est": round(max_hr, 0),
    }


def _strain_label(score: float) -> str:
    if score < 4:
        return "minimal"
    elif score < 8:
        return "light"
    elif score < 14:
        return "moderate"
    elif score < 18:
        return "high"
    else:
        return "overreaching"
