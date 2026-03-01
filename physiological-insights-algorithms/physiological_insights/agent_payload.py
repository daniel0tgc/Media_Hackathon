"""Build the lean agent_payload.json (<2K tokens) from the full analysis packet."""

import datetime
import numpy as np


def build_agent_payload(full_packet: dict, results: dict) -> dict:
    """Deterministically trim analysis_full.json into an agent-ready payload."""
    payload = {
        "schema_version": "3.0",
        "meta": _build_meta(full_packet),
        "agent_state": _build_agent_state(full_packet, results),
        "data_quality": _build_data_quality(full_packet, results),
        "alerts": _build_alerts(full_packet, results),
        "baseline": _build_baseline(full_packet, results),
        "latest_day": _trim_latest_day(full_packet),
        "circadian": _build_circadian(full_packet),
        "recovery": _build_recovery(full_packet, results),
        "sleep_debt": _build_sleep_debt(full_packet, results),
        "sleep_performance": _trim_sleep_performance(full_packet),
        "sleep_architecture_summary": _trim_arch_summary(full_packet),
        "strain": _build_strain(full_packet, results),
        "trends": full_packet.get("trends"),
    }
    return _strip_nulls(payload)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

def _build_meta(pkt: dict) -> dict:
    m = pkt.get("meta", {})
    return {
        "analysis_generated_at": m.get("analysis_generated_at"),
        "data_coverage": m.get("data_coverage"),
    }


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

def _build_agent_state(pkt: dict, results: dict) -> dict:
    recovery = pkt.get("recovery") or {}
    strain = results.get("strain", {})
    daily_strain = strain.get("daily_strain", [])
    sleep_debt = pkt.get("sleep_debt") or {}
    arch = _get_arch_summary(pkt)
    circ = pkt.get("circadian_profile") or {}

    zone = recovery.get("zone")
    history = recovery.get("history", [])
    days_since_green = recovery.get("days_since_green_zone")

    readiness_regime = _derive_readiness_regime(zone, history, days_since_green)
    consecutive_overreach = _count_overreach(daily_strain)
    days_since_rest = _days_since_rest(daily_strain)
    debt_hours = sleep_debt.get("current_debt_hours") or 0

    avg_rem_pct = arch.get("avg_rem_pct") if arch else None
    rem_deficit_alert = avg_rem_pct is not None and avg_rem_pct < 18

    bedtime_var = circ.get("bedtime_variance_min")
    circadian_anchor_missing = bedtime_var is not None and bedtime_var > 45

    recommended_ceiling = _strain_ceiling(zone, consecutive_overreach)
    deep_work = _deep_work_capacity(recommended_ceiling, rem_deficit_alert, debt_hours > 1.5)

    peak_window = circ.get("estimated_peak_window")
    nap_recommended = rem_deficit_alert and debt_hours > 1.5
    nap_window = _derive_nap_window(peak_window)

    perf = results.get("performance", {})
    daily_ready = perf.get("daily_ready", [])
    n_ready = sum(d.get("n", 0) for d in daily_ready) if daily_ready else 0
    peak_confidence = _peak_cognitive_confidence(n_ready, zone)

    return {
        "readiness_regime": readiness_regime,
        "consecutive_overreach_days": consecutive_overreach,
        "days_since_rest_day": days_since_rest,
        "sleep_debt_alert": debt_hours > 1.5,
        "rem_deficit_alert": rem_deficit_alert,
        "circadian_anchor_missing": circadian_anchor_missing,
        "recommended_strain_ceiling": recommended_ceiling,
        "deep_work_capacity": deep_work,
        "nap_recommended": nap_recommended,
        "nap_window_local": nap_window if nap_recommended else None,
        "peak_cognitive_window_local": peak_window,
        "peak_cognitive_confidence": peak_confidence,
    }


def _derive_readiness_regime(zone, history, days_since_green):
    if zone == "red":
        return "red"
    if zone == "yellow":
        if days_since_green is not None and days_since_green >= 3:
            return "chronic_yellow"
        return "yellow"
    return "green"


def _count_overreach(daily_strain: list) -> int:
    count = 0
    for d in reversed(daily_strain):
        if d.get("strain_score", 0) >= 18:
            count += 1
        else:
            break
    return count


def _days_since_rest(daily_strain: list) -> int | None:
    if not daily_strain:
        return None
    for i, d in enumerate(reversed(daily_strain)):
        if d.get("strain_score", 21) < 5:
            return i
    return len(daily_strain)


def _strain_ceiling(zone, overreach_count):
    if zone == "red":
        return 8
    if zone == "yellow":
        return max(8, 14 - overreach_count * 2)
    return max(10, 18 - overreach_count * 2)


def _deep_work_capacity(ceiling, rem_deficit, debt_alert):
    if ceiling <= 8 or (rem_deficit and debt_alert):
        return "low"
    if ceiling <= 12 or rem_deficit or debt_alert:
        return "moderate"
    return "high"


def _derive_nap_window(peak_window):
    if not peak_window:
        return "13:00-15:00"
    try:
        end_str = peak_window.split("-")[1].replace(":00", "")
        end_h = int(end_str)
        nap_h = min(end_h + 2, 16)
        return f"{nap_h:02d}:00-{nap_h + 1:02d}:00"
    except (IndexError, ValueError):
        return "13:00-15:00"


def _peak_cognitive_confidence(n_ready, zone):
    if n_ready >= 30 and zone in ("green", None):
        return "high"
    if n_ready >= 15:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------

def _build_data_quality(pkt: dict, results: dict) -> dict:
    coverage = pkt.get("meta", {}).get("data_coverage", {})

    sensor_days = 0
    if "sensor_data" in coverage:
        try:
            from_dt = datetime.datetime.fromisoformat(str(coverage["sensor_data"]["from"]).replace("Z", "+00:00"))
            to_dt = datetime.datetime.fromisoformat(str(coverage["sensor_data"]["to"]).replace("Z", "+00:00"))
            sensor_days = max(1, (to_dt - from_dt).days + 1)
        except Exception:
            sensor_days = 0

    ss = results.get("sleep_sessions", {})
    all_nights = ss.get("nights", [])
    primary_count = sum(1 for n in all_nights if n.get("session_type") == "primary")

    perf = results.get("performance", {})
    daily_sr = perf.get("daily_self_report", [])
    daily_ready = perf.get("daily_ready", [])
    n_ready = sum(d.get("n", 0) for d in daily_ready) if daily_ready else 0

    sr_populated = sum(1 for d in daily_sr if any(d.get(k) is not None for k in ("stress", "sleepiness", "sharpness")))
    sr_pct = round(sr_populated / len(daily_sr) * 100) if daily_sr else 0

    hrv_populated_pct = 0
    hrv_ts = results.get("hrv", {}).get("timeseries", [])
    if hrv_ts and all_nights:
        hrv_populated_pct = min(100, round(len(hrv_ts) / max(1, primary_count) * 100 / 50))

    strain_data = results.get("strain", {}).get("daily_strain", [])
    scores = [d["strain_score"] for d in strain_data if d.get("strain_score") is not None]
    strain_var = "none"
    if len(scores) >= 3:
        std = float(np.std(scores))
        strain_var = "high" if std > 5 else "moderate" if std > 2 else "low"
    elif len(scores) >= 2:
        strain_var = "low"

    circ = results.get("circadian", {})
    cosinor_n = circ.get("cosinor_n", 0)
    chronotype_conf = "high" if cosinor_n >= 30 else "medium" if cosinor_n >= 15 else "low"

    overall = _overall_confidence(sensor_days, primary_count, n_ready, sr_pct)

    quality = {
        "sensor_days": sensor_days,
        "sleep_sessions_primary": primary_count,
        "self_report_populated_pct": sr_pct,
        "morning_hrv_populated_pct": hrv_populated_pct,
        "focus_metric_valid": n_ready >= 30,
        "strain_recovery_curve_learnable": strain_var != "none",
        "chronotype_confidence": chronotype_conf,
        "overall_confidence": overall,
    }
    if overall == "low":
        quality["confidence_note"] = _confidence_note(sensor_days, primary_count, n_ready)
    return quality


def _overall_confidence(sensor_days, primary_count, n_ready, sr_pct):
    score = 0
    if sensor_days >= 7:
        score += 1
    if primary_count >= 5:
        score += 1
    if n_ready >= 30:
        score += 1
    if sr_pct >= 50:
        score += 1
    if score >= 3:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _confidence_note(sensor_days, primary_count, n_ready):
    parts = []
    if sensor_days < 7:
        parts.append(f"only {sensor_days} sensor day(s)")
    if primary_count < 5:
        parts.append(f"only {primary_count} primary sleep session(s)")
    if n_ready < 15:
        parts.append(f"only {n_ready} Ready test(s)")
    return "Low confidence: " + ", ".join(parts) + ". Collect more data for reliable insights."


# ---------------------------------------------------------------------------
# Structured Alerts
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _build_alerts(pkt: dict, results: dict) -> list[dict]:
    alerts = []
    recovery = pkt.get("recovery") or {}
    sleep_debt = pkt.get("sleep_debt") or {}
    arch = _get_arch_summary(pkt)
    strain = results.get("strain", {})
    daily_strain = strain.get("daily_strain", [])
    circ = pkt.get("circadian_profile") or {}

    # CHRONIC_UNDERRECOVERY
    zone = recovery.get("zone")
    dsg = recovery.get("days_since_green_zone")
    if zone in ("red", "yellow") and dsg is not None and dsg >= 3:
        sev = "critical" if zone == "red" else "warning"
        alerts.append({
            "id": "CHRONIC_UNDERRECOVERY",
            "severity": sev,
            "message": f"Recovery has not reached green zone in {dsg} sessions.",
            "recommended_action": "Prioritize sleep, reduce training load, and consider a rest day.",
        })

    # SLEEP_DEBT_ACCUMULATING
    debt_h = sleep_debt.get("current_debt_hours", 0) or 0
    trend = sleep_debt.get("trend")
    if debt_h > 1.5:
        sev = "critical" if debt_h > 4 else "warning"
        alerts.append({
            "id": "SLEEP_DEBT_ACCUMULATING",
            "severity": sev,
            "message": f"You owe {debt_h}h of sleep. Trend: {trend or 'unknown'}.",
            "recommended_action": f"Aim for {30 + int(debt_h * 15)} extra minutes tonight.",
        })

    # OVERREACH_STREAK
    overreach = _count_overreach(daily_strain)
    if overreach >= 2:
        alerts.append({
            "id": "OVERREACH_STREAK",
            "severity": "warning",
            "message": f"Strain ≥18 for {overreach} consecutive day(s). Risk of functional overreach.",
            "recommended_action": "Schedule a rest day. Reduce strain below 10.",
        })

    # REM_DEFICIT
    avg_rem = arch.get("avg_rem_pct") if arch else None
    rem_debt = arch.get("rem_debt_min_3night") if arch else None
    if avg_rem is not None and avg_rem < 18:
        alerts.append({
            "id": "REM_DEFICIT",
            "severity": "warning",
            "message": f"Average REM is {avg_rem}% (target ≥20%). REM debt over last 3 nights: {rem_debt or '?'} min.",
            "recommended_action": "Avoid alcohol and late caffeine. Aim for consistent wake time.",
        })

    # DEEP_SLEEP_DEFICIT
    avg_deep = arch.get("avg_deep_pct") if arch else None
    if avg_deep is not None and avg_deep < 15:
        alerts.append({
            "id": "DEEP_SLEEP_DEFICIT",
            "severity": "warning",
            "message": f"Average deep sleep is {avg_deep}% (target 15-20%).",
            "recommended_action": "Exercise earlier in the day. Keep bedroom cool (65-68°F).",
        })

    # CIRCADIAN_DRIFT
    bedtime_var = circ.get("bedtime_variance_min")
    if bedtime_var is not None and bedtime_var > 45:
        alerts.append({
            "id": "CIRCADIAN_DRIFT",
            "severity": "info",
            "message": f"Bedtime varies by {int(bedtime_var)} min. Circadian anchor is unstable.",
            "recommended_action": "Set a consistent bedtime within a 30-minute window.",
        })

    # REST_DAY_OVERDUE
    days_rest = _days_since_rest(daily_strain)
    if days_rest is not None and days_rest >= 7:
        alerts.append({
            "id": "REST_DAY_OVERDUE",
            "severity": "info",
            "message": f"No rest day (strain <5) in the last {days_rest} days.",
            "recommended_action": "Schedule a low-strain recovery day.",
        })

    alerts.sort(key=lambda a: _SEVERITY_ORDER.get(a["severity"], 9))
    return alerts


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def _build_baseline(pkt: dict, results: dict) -> dict | None:
    b = pkt.get("baseline")
    if not b:
        return None

    perf = results.get("performance", {})
    daily_ready = perf.get("daily_ready", [])
    scores = [d["mean"] for d in daily_ready if d.get("mean") is not None]

    out = {
        "ready_score": b.get("ready_score"),
        "agility_peak": b.get("agility_peak"),
        "rmssd_sleep_peak_ms": b.get("rmssd_sleep_peak_ms"),
    }
    if scores:
        arr = np.array(scores)
        out["ready_p25"] = round(float(np.percentile(arr, 25)), 1)
        out["ready_p50"] = round(float(np.percentile(arr, 50)), 1)
        out["ready_p75"] = round(float(np.percentile(arr, 75)), 1)

    n_focus = sum(1 for d in perf.get("daily_focus", []) if d.get("mean") is not None)
    if n_focus >= 30:
        out["focus_peak"] = b.get("focus_peak")

    return out


# ---------------------------------------------------------------------------
# Latest Day (trimmed)
# ---------------------------------------------------------------------------

def _trim_latest_day(pkt: dict) -> dict | None:
    ld = pkt.get("latest_day")
    if not ld:
        return None
    return {
        "date": ld.get("date"),
        "readiness_tier": ld.get("readiness_tier"),
        "ready_mean": ld.get("ready_mean"),
        "ready_pct_baseline": ld.get("ready_pct_baseline"),
        "agility_mean": ld.get("agility_mean"),
        "focus_mean": ld.get("focus_mean"),
        "self_report": ld.get("self_report"),
        "strain": ld.get("strain"),
    }


# ---------------------------------------------------------------------------
# Circadian (enhanced)
# ---------------------------------------------------------------------------

def _build_circadian(pkt: dict) -> dict | None:
    cp = pkt.get("circadian_profile")
    if not cp:
        return None
    circ_raw = {}
    # Merge circadian_profile from full packet
    for k in ("estimated_peak_window", "cosinor_acrophase_hour", "cosinor_amplitude", "worst_window"):
        circ_raw[k] = cp.get(k)
    # Enhanced fields are stored in analysis_full under circadian_profile since we added them
    for k in ("peak_score_variance", "natural_sleep_onset_local", "natural_wake_local",
              "bedtime_variance_min", "chronotype_estimate", "cosinor_n"):
        circ_raw[k] = cp.get(k)
    return circ_raw


# ---------------------------------------------------------------------------
# Recovery (enhanced)
# ---------------------------------------------------------------------------

def _build_recovery(pkt: dict, results: dict) -> dict | None:
    r = pkt.get("recovery")
    if not r or r.get("latest") is None:
        return None

    ss = results.get("sleep_sessions", {})
    nights = ss.get("nights", [])
    primary = [n for n in nights if n.get("session_type") == "primary"]

    limiting = _recovery_limiting_factor(r, pkt.get("sleep_debt"), primary)

    out = {
        "latest": r.get("latest"),
        "zone": r.get("zone"),
        "personal_mean": r.get("personal_mean"),
        "pct_of_mean": r.get("pct_of_mean"),
        "trend": r.get("trend"),
        "days_since_green_zone": r.get("days_since_green_zone"),
        "recovery_limiting_factor": limiting,
    }
    return out


def _recovery_limiting_factor(recovery, sleep_debt, primary_nights):
    factors = []
    if sleep_debt:
        debt_h = sleep_debt.get("current_debt_hours", 0) or 0
        if debt_h > 2:
            factors.append("sleep_debt")

    if primary_nights:
        last = primary_nights[-1] if primary_nights else {}
        if last.get("avg_hrv_rmssd_ms") is not None and last["avg_hrv_rmssd_ms"] < 30:
            factors.append("low_hrv")
        if last.get("efficiency_pct") is not None and last["efficiency_pct"] < 80:
            factors.append("poor_sleep_efficiency")
        if last.get("deep_pct") is not None and last["deep_pct"] < 10:
            factors.append("low_deep_sleep")

    if not factors:
        return None
    return factors[0]


# ---------------------------------------------------------------------------
# Sleep Debt (enhanced)
# ---------------------------------------------------------------------------

def _build_sleep_debt(pkt: dict, results: dict) -> dict | None:
    sd = pkt.get("sleep_debt")
    if not sd or sd.get("current_debt_min") is None:
        return None

    ss = results.get("sleep_sessions", {})
    arch = ss.get("architecture_summary") or {}
    avg_bedtime = arch.get("avg_bedtime_local")

    recommended_bt = None
    if avg_bedtime:
        try:
            parts = avg_bedtime.split(":")
            bt_h = int(parts[0])
            bt_m = int(parts[1])
            debt_h = sd.get("current_debt_hours", 0) or 0
            extra_min = min(60, int(debt_h * 15))
            total_min = bt_h * 60 + bt_m - extra_min
            if total_min < 0:
                total_min += 24 * 60
            recommended_bt = f"{total_min // 60:02d}:{total_min % 60:02d}"
        except (ValueError, IndexError):
            pass

    nightly_deficit_trend = sd.get("trend")

    return {
        "current_debt_min": sd.get("current_debt_min"),
        "current_debt_hours": sd.get("current_debt_hours"),
        "nights_to_repay": sd.get("nights_to_repay"),
        "avg_nightly_deficit_min": sd.get("avg_nightly_deficit_min"),
        "trend": sd.get("trend"),
        "recommended_bedtime_tonight_local": recommended_bt,
        "nightly_deficit_trend": nightly_deficit_trend,
    }


# ---------------------------------------------------------------------------
# Sleep Performance & Architecture (pass-through trimmed)
# ---------------------------------------------------------------------------

def _trim_sleep_performance(pkt: dict) -> dict | None:
    ss = pkt.get("sleep_sessions")
    if not ss:
        return None
    return ss.get("sleep_performance")


def _trim_arch_summary(pkt: dict) -> dict | None:
    return _get_arch_summary(pkt)


def _get_arch_summary(pkt: dict) -> dict | None:
    ss = pkt.get("sleep_sessions")
    if not ss:
        return None
    return ss.get("architecture_summary")


# ---------------------------------------------------------------------------
# Strain (enhanced)
# ---------------------------------------------------------------------------

def _build_strain(pkt: dict, results: dict) -> dict | None:
    strain_full = pkt.get("strain")
    if not strain_full:
        return None

    daily = strain_full.get("daily_strain", [])
    if not daily:
        return None

    latest = daily[-1]
    scores = [d["strain_score"] for d in daily]

    days_since_rest = _days_since_rest(daily)
    strain_var = "none"
    if len(scores) >= 3:
        std = float(np.std(scores))
        strain_var = "high" if std > 5 else "moderate" if std > 2 else "low"
    elif len(scores) >= 2:
        strain_var = "low"

    return {
        "latest_score": latest.get("strain_score"),
        "latest_level": latest.get("strain_level"),
        "latest_sleep_need_adj_min": latest.get("sleep_need_adjustment_min"),
        "max_hr_est": strain_full.get("max_hr_est"),
        "days_since_rest_day": days_since_rest,
        "strain_variability": strain_var,
        "rest_day_overdue": days_since_rest is not None and days_since_rest >= 7,
    }


# ---------------------------------------------------------------------------
# Null-stripping utility
# ---------------------------------------------------------------------------

def _strip_nulls(obj):
    """Recursively remove keys whose values are None or empty dicts/lists.
    Also converts numpy types to native Python types for JSON safety."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v = _strip_nulls(v)
            if v is None:
                continue
            if isinstance(v, dict) and not v:
                continue
            cleaned[k] = v
        return cleaned if cleaned else None
    if isinstance(obj, list):
        cleaned = [_strip_nulls(item) for item in obj]
        return [c for c in cleaned if c is not None]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
