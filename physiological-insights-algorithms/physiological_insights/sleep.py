"""Sleep detection: onset/offset, duration, HR nadir, wake episodes, continuity."""

import numpy as np
import pandas as pd


def _acc_energy(row: pd.Series) -> float:
    """Compute total accelerometer energy for an epoch using RMS columns if available."""
    rms_cols = [c for c in row.index if c.startswith("acc_") and "rms" in c.lower()]
    if rms_cols:
        vals = row[rms_cols].dropna()
        if not vals.empty:
            return float(np.sqrt((vals ** 2).sum()))
    energy_cols = [c for c in row.index if c.startswith("acc_") and "energyPerSec" in c]
    if energy_cols:
        vals = row[energy_cols].dropna()
        return float(vals.sum()) if not vals.empty else 0.0
    return 0.0


def analyse_sleep(df: pd.DataFrame) -> dict:
    """Detect sleep periods and compute metrics from decoded-metrics data."""
    result: dict = {"nights": []}

    wear = df[df["wear_mode"] == "wear_on"].copy()
    if wear.empty or "heart_rate_mean" not in wear.columns:
        return {
            "nights": [],
            "latest_night": None,
        }

    wear = wear.sort_values("datetime").reset_index(drop=True)

    # Compute accelerometer energy per epoch
    wear["acc_energy"] = wear.apply(_acc_energy, axis=1)

    # Group data by calendar night (6pm to 6pm next day to capture evening-to-morning sleep)
    wear["night_date"] = (wear["datetime_et"] - pd.Timedelta(hours=18)).dt.date

    nights = []
    for night_date, night_df in wear.groupby("night_date"):
        hr = night_df["heart_rate_mean"]
        rmssd = night_df.get("cardio_RMSSD_ms")

        # Sleep onset: first epoch where HR < 65 within expected sleep window (9pm-3am local)
        hour_of_day = night_df["datetime_et"].dt.hour
        sleep_window = night_df[(hour_of_day >= 21) | (hour_of_day <= 3)]

        if sleep_window.empty:
            continue

        onset_candidates = sleep_window[sleep_window["heart_rate_mean"] < 65]
        if rmssd is not None and not rmssd.isna().all():
            onset_candidates = onset_candidates[onset_candidates.get("cardio_RMSSD_ms", pd.Series(dtype=float)) > 70]

        if onset_candidates.empty:
            onset_candidates = sleep_window[sleep_window["heart_rate_mean"] < 70]

        if onset_candidates.empty:
            continue

        sleep_onset = onset_candidates["datetime_et"].iloc[0]

        # Sleep offset: last epoch meeting sleep criteria before sustained wake
        post_onset = night_df[night_df["datetime_et"] >= sleep_onset]
        wake_thresh_hr = 70
        wake_candidates = post_onset[post_onset["heart_rate_mean"] >= wake_thresh_hr]

        if not wake_candidates.empty:
            # Find first sustained wake (5+ consecutive epochs ~2.5 min at 30s epochs)
            wake_idx = wake_candidates.index
            consecutive = 1
            sleep_offset_idx = wake_idx[0]
            for i in range(1, len(wake_idx)):
                if wake_idx[i] == wake_idx[i - 1] + 1:
                    consecutive += 1
                    if consecutive >= 5:
                        sleep_offset_idx = wake_idx[i - consecutive + 1]
                        break
                else:
                    consecutive = 1

            sleep_offset = post_onset.loc[sleep_offset_idx, "datetime_et"]
        else:
            sleep_offset = post_onset["datetime_et"].iloc[-1]

        duration_min = (sleep_offset - sleep_onset).total_seconds() / 60
        if duration_min < 60:
            continue

        # HR nadir: minimum 30-min rolling mean during sleep
        sleep_period = night_df[
            (night_df["datetime_et"] >= sleep_onset)
            & (night_df["datetime_et"] <= sleep_offset)
        ]
        hr_rolling = sleep_period["heart_rate_mean"].rolling(60, min_periods=10).mean()
        hr_nadir = float(hr_rolling.min()) if not hr_rolling.isna().all() else None

        # Wake episodes: 5-min windows during sleep where HR > 70 AND acc_energy > 500
        wake_episodes = 0
        if not sleep_period.empty:
            high_hr = sleep_period["heart_rate_mean"] > 70
            high_motion = sleep_period["acc_energy"] > 500
            wake_mask = high_hr & high_motion
            wake_epochs = sleep_period[wake_mask]
            # Count distinct clusters (gap of 5+ epochs = new episode)
            if not wake_epochs.empty:
                idx_arr = wake_epochs.index.values
                gaps = np.diff(idx_arr) > 5
                wake_episodes = int(np.sum(gaps)) + 1

        continuity = round(1 - (wake_episodes * 5 / max(duration_min, 1)), 3)
        continuity = max(0.0, min(1.0, continuity))

        night_result = {
            "night_date": str(night_date),
            "sleep_onset": str(sleep_onset),
            "sleep_offset": str(sleep_offset),
            "duration_min": round(duration_min, 1),
            "hr_nadir_bpm": round(hr_nadir, 1) if hr_nadir else None,
            "wake_episodes": wake_episodes,
            "continuity_score": continuity,
        }
        nights.append(night_result)

    result["nights"] = nights
    result["latest_night"] = nights[-1] if nights else None

    return result
