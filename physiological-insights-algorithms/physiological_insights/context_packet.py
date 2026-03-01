"""Assemble all module outputs into the full analysis_output.json structure."""

import datetime
import pandas as pd


def _safe(val, precision=1):
    """Safely round numeric values, pass through None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return round(float(val), precision)
    except (TypeError, ValueError):
        return val


def _build_daily_summaries(tests_df, results: dict) -> list[dict]:
    """Build per-day summary dicts from test results and analysis outputs."""
    perf = results.get("performance", {})
    if not perf:
        return []

    readiness = results.get("readiness", {})
    tiers = readiness.get("tiers_by_date", {})
    daily_sr = {str(d.get("date", ""))[:10]: d for d in perf.get("daily_self_report", [])}

    activity = results.get("activity", {})
    daily_activity = {d["date"]: d for d in activity.get("daily_activity", [])}

    strain_data = results.get("strain", {})
    daily_strain = {d["date"]: d for d in strain_data.get("daily_strain", [])}

    ready_by_date = {str(d.get("date", ""))[:10]: d for d in perf.get("daily_ready", [])}
    agility_by_date = {str(d.get("date", ""))[:10]: d for d in perf.get("daily_agility", [])}
    focus_by_date = {str(d.get("date", ""))[:10]: d for d in perf.get("daily_focus", [])}

    all_dates = sorted(set(
        list(ready_by_date.keys()) +
        list(agility_by_date.keys()) +
        list(focus_by_date.keys())
    ))

    baseline = perf.get("ready_baseline")
    agility_peak = perf.get("agility", {}).get("peak")

    summaries = []
    for date_str in all_dates:
        rd = ready_by_date.get(date_str, {})
        ad = agility_by_date.get(date_str, {})
        fd = focus_by_date.get(date_str, {})
        sr = daily_sr.get(date_str, {})
        tier_info = tiers.get(date_str, {})
        act = daily_activity.get(date_str, {})
        st = daily_strain.get(date_str, {})

        ready_mean = rd.get("mean")
        agility_mean = ad.get("mean")

        day_tests = tests_df[tests_df["date"].astype(str) == date_str] if tests_df is not None else pd.DataFrame()
        notes = []
        if "context_note" in day_tests.columns:
            notes = [n for n in day_tests["context_note"].dropna().unique() if n.strip()]

        summary = {
            "date": date_str,
            "readiness_tier": tier_info.get("tier", "Unknown"),
            "ready_mean": _safe(ready_mean),
            "ready_pct_baseline": _safe(ready_mean / baseline * 100) if ready_mean and baseline else None,
            "agility_mean": _safe(agility_mean),
            "agility_pct_peak": _safe(agility_mean / agility_peak * 100) if agility_mean and agility_peak else None,
            "focus_mean": _safe(fd.get("mean")),
            "self_report": {
                "stress": _safe(sr.get("stress")),
                "sleepiness": _safe(sr.get("sleepiness")),
                "sharpness": _safe(sr.get("sharpness")),
            },
            "hrv": {
                "morning_rmssd_ms": None,
                "rmssd_pct_baseline": None,
            },
            "sleep": {
                "duration_min": None,
                "wake_episodes": None,
                "hr_nadir_bpm": None,
            },
            "activity": {
                "exercise_load": act.get("exercise_load"),
                "steps": act.get("steps", 0),
                "calories": _safe(act.get("calories", 0)),
            },
            "strain": {
                "score": st.get("strain_score"),
                "level": st.get("strain_level"),
                "sleep_need_adjustment_min": st.get("sleep_need_adjustment_min"),
            } if st else None,
            "context_notes": notes[:5],
        }
        summaries.append(summary)

    sleep_data = results.get("sleep", {})
    for night in sleep_data.get("nights", []):
        night_date = night.get("night_date")
        for s in summaries:
            if s["date"] == night_date:
                s["sleep"]["duration_min"] = _safe(night.get("duration_min"))
                s["sleep"]["wake_episodes"] = night.get("wake_episodes")
                s["sleep"]["hr_nadir_bpm"] = _safe(night.get("hr_nadir_bpm"))
                break

    return summaries


def _build_weekly_summaries(results: dict) -> list[dict]:
    """Shape weekly data from performance module output."""
    perf = results.get("performance", {})
    if not perf:
        return []

    weekly_raw = perf.get("weekly", [])
    baseline = perf.get("ready_baseline")

    weeks = []
    prev_week = None
    for w in weekly_raw:
        entry = {
            "week_label": w.get("week_label"),
            "date_range": {
                "from": str(w.get("date_min", ""))[:10],
                "to": str(w.get("date_max", ""))[:10],
            },
            "ready_week_mean": _safe(w.get("ready_mean")),
            "ready_week_pct_baseline": _safe(w.get("ready_pct_baseline")),
            "agility_week_mean": _safe(w.get("agility_mean")),
            "focus_week_mean": _safe(w.get("focus_mean")),
            "self_report_means": {
                "stress": _safe(w.get("stress")),
                "sleepiness": _safe(w.get("sleepiness")),
                "sharpness": _safe(w.get("sharpness")),
            },
            "test_count": w.get("test_count", 0),
        }

        if prev_week:
            prev_ready = prev_week.get("ready_mean", 0) or 1
            current_ready = w.get("ready_mean", 0) or 0
            ready_change = (current_ready - prev_ready) / prev_ready * 100 if prev_ready else None

            prev_ag = prev_week.get("agility_mean")
            cur_ag = w.get("agility_mean")
            ag_change = ((cur_ag - prev_ag) / prev_ag * 100) if prev_ag and cur_ag else None

            prev_stress = prev_week.get("stress")
            cur_stress = w.get("stress")
            stress_change = (cur_stress - prev_stress) if prev_stress is not None and cur_stress is not None else None

            entry["vs_prior_week"] = {
                "ready_change_pct": _safe(ready_change),
                "agility_change_pct": _safe(ag_change),
                "stress_change": _safe(stress_change),
            }

        weeks.append(entry)
        prev_week = w

    return weeks


def _build_sleep_session_section(results: dict) -> dict | None:
    """Build the sleep sessions section from sleep_sessions analysis."""
    ss = results.get("sleep_sessions")
    if not ss:
        return None
    return {
        "nights": ss.get("nights", []),
        "architecture_summary": ss.get("architecture_summary"),
        "sleep_performance": ss.get("sleep_performance"),
    }


def _build_sleep_debt_section(results: dict) -> dict | None:
    ss = results.get("sleep_sessions")
    if not ss:
        return None
    return ss.get("sleep_debt")


def _build_recovery_section(results: dict) -> dict | None:
    ss = results.get("sleep_sessions")
    if not ss:
        return None
    return ss.get("recovery")


def _build_strain_section(results: dict) -> dict | None:
    strain = results.get("strain")
    if not strain or not strain.get("daily_strain"):
        return None
    return strain


def _build_insights(results: dict, packet: dict) -> list[str]:
    """Generate plain-English Whoop-level insight strings from analysis results."""
    insights = []
    perf = results.get("performance", {})
    baseline = perf.get("ready_baseline")
    circ = results.get("circadian", {})
    ss = results.get("sleep_sessions", {})
    strain = results.get("strain", {})

    # Ready vs baseline
    daily_ready = perf.get("daily_ready", [])
    if daily_ready and baseline:
        latest_mean = daily_ready[-1].get("mean")
        if latest_mean:
            pct = latest_mean / baseline * 100
            insights.append(
                f"Ready baseline is {baseline}. Current average is {latest_mean:.1f} ({pct:.1f}% of baseline)."
            )

    # Weekly trend
    weekly = packet.get("weekly_summaries", [])
    if len(weekly) >= 2:
        vs = weekly[-1].get("vs_prior_week", {})
        change = vs.get("ready_change_pct")
        if change:
            insights.append(
                f"Week {weekly[-1]['week_label']} showed a {abs(change):.1f}% Ready {'decline' if change < 0 else 'improvement'} vs {weekly[-2]['week_label']}."
            )

    # Circadian peak
    peak = circ.get("estimated_peak_window")
    if peak:
        insights.append(f"Best performance window is {peak} (local time).")

    # Ready slope
    slope = perf.get("ready_7d_slope")
    if slope is not None:
        direction = "declining" if slope < 0 else "improving" if slope > 0 else "flat"
        insights.append(f"7-day Ready trend is {direction} (slope = {slope:.1f} points/day).")

    # --- Whoop-level sleep insights ---
    debt = ss.get("sleep_debt")
    if debt and debt.get("current_debt_hours") is not None:
        debt_h = debt["current_debt_hours"]
        if debt_h > 0:
            days = debt.get("nights_to_repay", "?")
            insights.append(
                f"You owe {debt_h} hours of sleep. At current recovery rate, this will take ~{days} days to repay."
            )
        else:
            insights.append("Sleep debt is fully repaid — you're in the green.")

    arch = ss.get("architecture_summary")
    if arch:
        deep_pct = arch.get("avg_deep_pct")
        rem_pct = arch.get("avg_rem_pct")
        if deep_pct is not None and deep_pct < 15:
            insights.append(
                f"Deep sleep averaged {deep_pct}% — below the 15-20% target. Expect reduced cognitive consolidation."
            )
        elif deep_pct is not None and deep_pct > 20:
            insights.append(f"Deep sleep averaged {deep_pct}% — above target. Excellent physical recovery.")

        if rem_pct is not None and rem_pct < 20:
            insights.append(
                f"REM sleep averaged {rem_pct}% (below 20-25% target) — may affect emotional regulation and memory."
            )

    recovery = ss.get("recovery")
    if recovery and recovery.get("latest") is not None:
        latest_rec = recovery["latest"]
        zone = recovery.get("zone", "")
        zone_label = {"green": "high", "yellow": "moderate", "red": "low"}.get(zone, "")
        rec_advice = {
            "green": "You're well recovered for high-intensity activity.",
            "yellow": "Light to moderate activity recommended; defer high-intensity training.",
            "red": "Rest strongly recommended. Your body needs recovery time.",
        }.get(zone, "")
        insights.append(
            f"Recovery score is {latest_rec}% ({zone_label}). {rec_advice}"
        )
        if recovery.get("pct_of_mean"):
            insights.append(
                f"Recovery is at {recovery['pct_of_mean']}% of your personal average ({recovery['personal_mean']}%)."
            )

    perf_score = ss.get("sleep_performance")
    if perf_score and perf_score.get("consistency") is not None:
        consistency = perf_score["consistency"]
        if consistency < 75:
            insights.append(
                f"Sleep consistency is {consistency}% — aim for more regular bed/wake times (<45 min variation)."
            )
        else:
            insights.append(f"Sleep consistency is {consistency}% — solid routine.")

    nights = ss.get("nights", [])
    for n in nights[-1:]:
        cc = n.get("circadian_compliance")
        if cc is not None:
            if cc >= 90:
                insights.append(f"Circadian compliance was {cc}% — sleep timing aligns well with your biological clock.")
            elif cc < 50:
                insights.append(f"Circadian compliance was only {cc}% — consider adjusting sleep schedule to align with circadian rhythm.")

    # --- Strain insights ---
    daily_strain = strain.get("daily_strain", [])
    if daily_strain:
        latest_strain = daily_strain[-1]
        s_score = latest_strain.get("strain_score", 0)
        s_level = latest_strain.get("strain_level", "")
        adj = latest_strain.get("sleep_need_adjustment_min", 0)
        if s_score >= 14:
            insights.append(
                f"Yesterday's strain was {s_score} ({s_level}) — your sleep need tonight is increased by ~{adj} minutes above baseline."
            )
        elif s_score > 0:
            insights.append(f"Recent strain was {s_score} ({s_level}).")

    # Pattern-derived insights
    for p in results.get("patterns", []):
        desc = p.get("description", "")
        if desc and desc not in insights:
            insights.append(desc)

    return insights


def build_context_packet(tests_df, metrics_df, sleep_df, results: dict, graphs_dir: str) -> dict:
    """Assemble the full analysis_output.json structure."""
    perf = results.get("performance", {})
    hrv = results.get("hrv", {})
    circ = results.get("circadian", {})
    sleep = results.get("sleep", {})
    readiness = results.get("readiness", {})

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    meta: dict = {
        "analysis_generated_at": now,
        "data_coverage": {},
        "graphs_directory": graphs_dir,
    }

    if tests_df is not None:
        meta["data_coverage"]["test_results"] = {
            "from": str(tests_df["created_at"].min()),
            "to": str(tests_df["created_at"].max()),
            "total_tests": len(tests_df),
        }
        user_ids = tests_df["user_id"].dropna().unique() if "user_id" in tests_df.columns else []
        if len(user_ids):
            meta["user_id"] = str(user_ids[0])

    if metrics_df is not None:
        meta["data_coverage"]["sensor_data"] = {
            "from": str(metrics_df["datetime"].min()),
            "to": str(metrics_df["datetime"].max()),
            "total_epochs": len(metrics_df),
        }

    if sleep_df is not None:
        meta["data_coverage"]["sleep_sessions"] = {
            "from": str(sleep_df["sleep_start"].min()),
            "to": str(sleep_df["sleep_start"].max()),
            "total_sessions": len(sleep_df),
        }

    # Baseline
    baseline = {
        "ready_score": perf.get("ready_baseline"),
        "agility_peak": perf.get("agility", {}).get("peak"),
        "focus_peak": perf.get("focus", {}).get("peak"),
        "rmssd_sleep_peak_ms": _safe(hrv.get("rmssd_sleep_peak")),
        "hr_nadir_bpm": None,
    }

    latest_night = sleep.get("latest_night")
    if latest_night:
        baseline["hr_nadir_bpm"] = _safe(latest_night.get("hr_nadir_bpm"))

    daily_summaries = _build_daily_summaries(tests_df, results)
    weekly_summaries = _build_weekly_summaries(results)

    latest_day = daily_summaries[-1] if daily_summaries else None

    trends = {
        "ready_7d_slope": _safe(perf.get("ready_7d_slope")),
        "ready_trend_direction": "declining" if (perf.get("ready_7d_slope") or 0) < -0.5 else "improving" if (perf.get("ready_7d_slope") or 0) > 0.5 else "stable",
        "rmssd_7d_slope": _safe(hrv.get("rmssd_7d_slope")),
        "agility_trend": "stable",
    }

    circadian_profile = {
        "estimated_peak_window": circ.get("estimated_peak_window"),
        "cosinor_acrophase_hour": _safe(circ.get("acrophase_hour")),
        "cosinor_amplitude": _safe(circ.get("amplitude")),
        "worst_window": circ.get("worst_window"),
        "cosinor_n": circ.get("cosinor_n"),
        "peak_score_variance": _safe(circ.get("peak_score_variance")),
        "natural_sleep_onset_local": circ.get("natural_sleep_onset_local"),
        "natural_wake_local": circ.get("natural_wake_local"),
        "bedtime_variance_min": _safe(circ.get("bedtime_variance_min"), 0),
        "chronotype_estimate": circ.get("chronotype_estimate"),
    }

    latest_tier = readiness.get("latest_tier", {})
    task_matching = latest_tier.get("task_suitability", {}) if latest_tier else {}

    graph_manifest = [
        {"filename": "ready_score_trajectory.png", "description": "Daily Ready score with 7-day rolling avg and baseline"},
        {"filename": "agility_focus_trajectory.png", "description": "Agility and Focus scores over time"},
        {"filename": "circadian_performance_curve.png", "description": "Ready scores by hour with cosinor fit"},
        {"filename": "self_report_vs_ready.png", "description": "Stress/sleepiness/sharpness vs Ready correlations"},
        {"filename": "hrv_night_profile.png", "description": "HR, RMSSD, and motion during captured night(s)"},
        {"filename": "weekly_readiness_summary.png", "description": "Week-over-week readiness comparison"},
        {"filename": "score_distributions.png", "description": "Histograms of Ready, Agility, Focus scores"},
        {"filename": "stress_sleepiness_heatmap.png", "description": "Heatmap of stress x sleepiness vs mean Ready"},
        {"filename": "sleep_architecture.png", "description": "Stacked bar chart of sleep stages per night"},
        {"filename": "sleep_debt_tracker.png", "description": "Running sleep debt with actual vs needed bars"},
        {"filename": "recovery_trend.png", "description": "Recovery score trend with color-coded zones"},
        {"filename": "strain_vs_recovery.png", "description": "Daily strain vs next-day recovery scatter"},
    ]

    packet = {
        "meta": meta,
        "baseline": baseline,
        "latest_day": latest_day,
        "daily_summaries": daily_summaries,
        "weekly_summaries": weekly_summaries,
        "trends": trends,
        "patterns_detected": results.get("patterns", []),
        "circadian_profile": circadian_profile,
        "task_matching": task_matching,
        "sleep_sessions": _build_sleep_session_section(results),
        "sleep_debt": _build_sleep_debt_section(results),
        "recovery": _build_recovery_section(results),
        "strain": _build_strain_section(results),
        "graphs": graph_manifest,
    }

    packet["insights"] = _build_insights(results, packet)

    return packet
