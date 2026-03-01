"""Multi-day pattern detection across physiological and performance signals."""

import pandas as pd
import numpy as np


def detect_patterns(tests_df: pd.DataFrame | None, results: dict) -> list[dict]:
    """Scan for clinically relevant multi-day patterns."""
    patterns: list[dict] = []
    perf = results.get("performance", {})
    hrv = results.get("hrv", {})
    sleep = results.get("sleep", {})
    sleep_sess = results.get("sleep_sessions", {})
    strain_data = results.get("strain", {})

    # --- 1. Compounding readiness debt: Ready declining over 5+ days ---
    daily_ready = perf.get("daily_ready", [])
    if len(daily_ready) >= 5:
        recent = daily_ready[-7:]
        means = [d["mean"] for d in recent if d.get("mean") is not None]
        if len(means) >= 5:
            deltas = [means[i + 1] - means[i] for i in range(len(means) - 1)]
            declining_days = sum(1 for d in deltas if d < 0)
            if declining_days >= 3:
                total_drop = means[-1] - means[0]
                baseline = perf.get("ready_baseline")
                pct_drop = abs(total_drop / baseline * 100) if baseline else None
                patterns.append({
                    "pattern": "compounding_readiness_debt",
                    "description": f"Ready scores declined {abs(total_drop):.0f} points ({pct_drop:.1f}%) over {len(means)} days" if pct_drop else f"Ready scores declined {abs(total_drop):.0f} points over {len(means)} days",
                    "severity": "warning",
                    "first_detected": str(recent[-1].get("date", ""))[:10],
                })

    # --- 2. RMSSD declining 3+ days ---
    rmssd_slope = hrv.get("rmssd_7d_slope")
    if rmssd_slope is not None and rmssd_slope < -2:
        patterns.append({
            "pattern": "rmssd_declining",
            "description": f"RMSSD trending down (slope={rmssd_slope:.1f} ms/day). Possible overreach.",
            "severity": "warning",
            "first_detected": None,
        })

    # --- 3. Sleep fragmentation (wake episodes > 4 for 2+ nights) ---
    nights = sleep.get("nights", [])
    fragmented_nights = [n for n in nights if (n.get("wake_episodes") or 0) > 4]
    if len(fragmented_nights) >= 2:
        patterns.append({
            "pattern": "sleep_fragmentation",
            "description": f"Wake episodes >4 on {len(fragmented_nights)} nights — sleep quality compromised",
            "severity": "warning",
            "first_detected": fragmented_nights[0].get("night_date"),
        })

    # --- 4. Short sleep (< 360 min / 6h for 2+ nights) ---
    short_nights = [n for n in nights if (n.get("duration_min") or 999) < 360]
    if len(short_nights) >= 2:
        patterns.append({
            "pattern": "sleep_debt",
            "description": f"Sleep <6h on {len(short_nights)} nights — accumulated sleep debt",
            "severity": "warning",
            "first_detected": short_nights[0].get("night_date"),
        })

    # --- 5. Stress + sleepiness compounding ---
    if tests_df is not None and "stress" in tests_df.columns and "sleepiness" in tests_df.columns:
        compound = tests_df[(tests_df["stress"] > 5) & (tests_df["sleepiness"] > 5)]
        if len(compound) >= 3:
            mean_ready = compound.loc[compound["type"] == "READY", "score"].mean()
            ready_note = f", correlated with Ready <{mean_ready:.0f}" if not pd.isna(mean_ready) else ""
            patterns.append({
                "pattern": "stress_sleepiness_compounding",
                "description": f"Stress >5 and sleepiness >5 simultaneously on {len(compound)} occasions{ready_note}",
                "severity": "warning",
                "first_detected": str(compound["date"].min()),
            })

    # --- 6. Subjective-objective mismatch ---
    if tests_df is not None and "sharpness" in tests_df.columns:
        ready_tests = tests_df[(tests_df["type"] == "READY") & tests_df["sharpness"].notna()].copy()
        if not ready_tests.empty:
            high_sharp_low_score = ready_tests[(ready_tests["sharpness"] >= 7) & (ready_tests["score"] < 140)]
            if not high_sharp_low_score.empty:
                worst = high_sharp_low_score.sort_values("score").iloc[0]
                patterns.append({
                    "pattern": "subjective_objective_mismatch",
                    "description": f"Self-reported sharpness {worst['sharpness']:.0f}/10 co-occurred with Ready {worst['score']:.0f} on {str(worst['date'])}",
                    "severity": "info",
                    "first_detected": str(worst["date"]),
                })

    # --- 7. Agility-Ready divergence ---
    weekly = perf.get("weekly", [])
    if len(weekly) >= 2:
        latest = weekly[-1]
        prior = weekly[-2]
        ready_change = latest.get("ready_change_pct")
        ag_current = latest.get("agility_mean")
        ag_prior = prior.get("agility_mean")
        if ready_change is not None and ag_current is not None and ag_prior is not None and ag_prior > 0:
            agility_change = (ag_current - ag_prior) / ag_prior * 100
            if ready_change < -5 and agility_change > 5:
                patterns.append({
                    "pattern": "agility_ready_divergence",
                    "description": f"Ready declined {abs(ready_change):.1f}% while Agility improved {agility_change:.1f}% — neuromuscular preserved but composite readiness degrading",
                    "severity": "info",
                    "first_detected": str(latest.get("date_min", ""))[:10],
                })

    # --- 8. Post-meal performance dip detection ---
    if tests_df is not None and "context_note" in tests_df.columns:
        meal_keywords = ["eating", "meal", "ate", "lunch", "dinner", "food", "snack"]
        meal_mask = tests_df["context_note"].str.lower().apply(
            lambda x: any(kw in str(x) for kw in meal_keywords) if pd.notna(x) else False
        )
        meal_tests = tests_df[meal_mask & (tests_df["type"] == "READY")]
        if len(meal_tests) >= 2:
            baseline = perf.get("ready_baseline")
            if baseline:
                mean_postmeal = meal_tests["score"].mean()
                dip = baseline - mean_postmeal
                if dip > 10:
                    patterns.append({
                        "pattern": "post_meal_dip_detected",
                        "description": f"Post-meal Ready averages {mean_postmeal:.0f} ({dip:.0f} points below baseline)",
                        "severity": "info",
                        "first_detected": str(meal_tests["date"].min()),
                    })

    # =========================================================================
    # Sleep-session-aware patterns (from fatigue CSV)
    # =========================================================================
    ss_nights = sleep_sess.get("nights", [])

    # --- 9. Chronic sleep debt ---
    debt_info = sleep_sess.get("sleep_debt")
    if debt_info and debt_info.get("current_debt_min") is not None:
        debt_min = debt_info["current_debt_min"]
        if debt_min > 120:
            debt_h = round(debt_min / 60, 1)
            patterns.append({
                "pattern": "chronic_sleep_debt",
                "description": f"Sleep debt is {debt_h}h — you owe significant sleep. Recovery takes ~{int(debt_h * 4)} days at current pace.",
                "severity": "warning",
                "first_detected": ss_nights[0].get("night_date") if ss_nights else None,
            })

    # --- 10. Deep sleep deficit ---
    low_deep = [n for n in ss_nights if n.get("deep_pct", 100) < 15 and n.get("total_sleep_min", 0) > 60]
    if len(low_deep) >= 2:
        avg_deep = np.mean([n["deep_pct"] for n in low_deep])
        patterns.append({
            "pattern": "deep_sleep_deficit",
            "description": f"Deep sleep below 15% target on {len(low_deep)} nights (avg {avg_deep:.1f}%). Cognitive consolidation and physical recovery may be impaired.",
            "severity": "warning",
            "first_detected": low_deep[0].get("night_date"),
        })

    # --- 11. REM sleep deficit ---
    low_rem = [n for n in ss_nights if n.get("rem_pct", 100) < 20 and n.get("total_sleep_min", 0) > 60]
    if len(low_rem) >= 2:
        avg_rem = np.mean([n["rem_pct"] for n in low_rem])
        patterns.append({
            "pattern": "rem_sleep_deficit",
            "description": f"REM sleep below 20% target on {len(low_rem)} nights (avg {avg_rem:.1f}%). May affect emotional regulation and memory.",
            "severity": "warning",
            "first_detected": low_rem[0].get("night_date"),
        })

    # --- 12. Low recovery ---
    low_recovery = [n for n in ss_nights if n.get("recovery_score") is not None and n["recovery_score"] < 34]
    if len(low_recovery) >= 2:
        patterns.append({
            "pattern": "low_recovery",
            "description": f"Recovery score in red zone (<34%) on {len(low_recovery)} sessions. Rest and light activity strongly recommended.",
            "severity": "warning",
            "first_detected": low_recovery[0].get("night_date"),
        })

    # --- 13. Sleep efficiency low ---
    low_eff = [n for n in ss_nights if n.get("efficiency_pct", 100) < 85 and n.get("session_time_min", 0) > 60]
    if len(low_eff) >= 2:
        avg_eff = np.mean([n["efficiency_pct"] for n in low_eff])
        patterns.append({
            "pattern": "sleep_efficiency_low",
            "description": f"Sleep efficiency below 85% on {len(low_eff)} nights (avg {avg_eff:.1f}%). Significant time in bed spent awake.",
            "severity": "info",
            "first_detected": low_eff[0].get("night_date"),
        })

    # --- 14. Circadian misalignment ---
    misaligned = [n for n in ss_nights if n.get("circadian_compliance") is not None and n["circadian_compliance"] < 50]
    if len(misaligned) >= 2:
        patterns.append({
            "pattern": "circadian_misalignment",
            "description": f"Circadian compliance below 50% on {len(misaligned)} nights. Sleep timing is misaligned with biological clock.",
            "severity": "warning",
            "first_detected": misaligned[0].get("night_date"),
        })

    return patterns
