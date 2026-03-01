"""Readiness tier assignment and task-type matching matrix."""


# Tier thresholds: (rmssd_pct_baseline_range, ready_score_range) -> tier
# When HRV is unavailable, only Ready score + self-report used
_TIERS = [
    {"name": "Peak",     "color": "green",  "rmssd_min": 90,  "ready_min": 175},
    {"name": "Good",     "color": "blue",   "rmssd_min": 75,  "ready_min": 155},
    {"name": "Moderate", "color": "yellow", "rmssd_min": 55,  "ready_min": 140},
    {"name": "Low",      "color": "orange", "rmssd_min": 40,  "ready_min": 130},
    {"name": "Recovery", "color": "red",    "rmssd_min": 0,   "ready_min": 0},
]

# Task suitability per tier
_TASK_MATRIX = {
    "Peak": {
        "deep_analytical_work": True,
        "creative_brainstorming": True,
        "high_stakes_presentation": True,
        "routine_admin": True,
        "physical_training": True,
    },
    "Good": {
        "deep_analytical_work": True,
        "creative_brainstorming": True,
        "high_stakes_presentation": True,
        "routine_admin": True,
        "physical_training": True,
    },
    "Moderate": {
        "deep_analytical_work": False,
        "creative_brainstorming": True,
        "high_stakes_presentation": False,
        "routine_admin": True,
        "physical_training": True,
    },
    "Low": {
        "deep_analytical_work": False,
        "creative_brainstorming": False,
        "high_stakes_presentation": False,
        "routine_admin": True,
        "physical_training": False,
    },
    "Recovery": {
        "deep_analytical_work": False,
        "creative_brainstorming": False,
        "high_stakes_presentation": False,
        "routine_admin": False,
        "physical_training": False,
    },
}


def _classify_tier(ready_score: float | None, rmssd_pct: float | None, stress: float | None, sleepiness: float | None) -> str:
    """Assign a readiness tier based on available signals."""
    # Severe self-report override: if both stress and sleepiness >7, cap at Low
    sr_cap = None
    if stress is not None and sleepiness is not None:
        if stress > 7 and sleepiness > 7:
            sr_cap = "Recovery"
        elif stress > 5 and sleepiness > 5:
            sr_cap = "Low"

    if ready_score is None:
        tier = "Moderate"  # fallback
    else:
        tier = "Recovery"
        for t in _TIERS:
            if rmssd_pct is not None:
                if rmssd_pct >= t["rmssd_min"] and ready_score >= t["ready_min"]:
                    tier = t["name"]
                    break
            else:
                if ready_score >= t["ready_min"]:
                    tier = t["name"]
                    break

    # Apply self-report cap (cannot promote, only demote)
    if sr_cap:
        tier_order = [t["name"] for t in _TIERS]
        if tier_order.index(tier) < tier_order.index(sr_cap):
            tier = sr_cap

    return tier


def assign_readiness_tiers(results: dict) -> dict:
    """Compute readiness tier for the latest day and historical days."""
    perf = results.get("performance", {})
    hrv = results.get("hrv", {})

    baseline = perf.get("ready_baseline")
    rmssd_pct = hrv.get("rmssd_pct_baseline")

    # Latest day tier
    daily_ready = perf.get("daily_ready", [])
    daily_sr = perf.get("daily_self_report", [])

    tiers_by_date: dict = {}

    if daily_ready:
        for day in daily_ready:
            date = str(day.get("date", ""))[:10]
            score = day.get("mean")
            stress = None
            sleepiness = None
            for sr in daily_sr:
                if str(sr.get("date", ""))[:10] == date:
                    stress = sr.get("stress")
                    sleepiness = sr.get("sleepiness")
                    break
            tier = _classify_tier(score, rmssd_pct, stress, sleepiness)
            tiers_by_date[date] = {
                "tier": tier,
                "color": next(t["color"] for t in _TIERS if t["name"] == tier),
                "ready_score": score,
                "task_suitability": _TASK_MATRIX.get(tier, {}),
            }

    latest_tier = None
    if daily_ready:
        last_date = str(daily_ready[-1].get("date", ""))[:10]
        latest_tier = tiers_by_date.get(last_date)

    return {
        "tiers_by_date": tiers_by_date,
        "latest_tier": latest_tier,
        "task_matrix": _TASK_MATRIX,
    }
