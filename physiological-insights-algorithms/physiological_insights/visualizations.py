"""Generate all analysis graphs as PNGs."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


_STYLE = {
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}

_TIER_COLORS = {
    "Peak": "#2ecc71",
    "Good": "#3498db",
    "Moderate": "#f1c40f",
    "Low": "#e67e22",
    "Recovery": "#e74c3c",
}

_TIER_BOUNDS = [
    ("Peak", 175, 200),
    ("Good", 155, 175),
    ("Moderate", 140, 155),
    ("Low", 130, 140),
    ("Recovery", 100, 130),
]


def generate_all_graphs(tests_df, metrics_df, sleep_df, results: dict, packet: dict, graphs_dir: str):
    plt.rcParams.update(_STYLE)
    os.makedirs(graphs_dir, exist_ok=True)

    perf = results.get("performance", {})
    strain = results.get("strain", {})
    ss = results.get("sleep_sessions", {})

    if tests_df is not None:
        _ready_score_with_overlays(tests_df, results, graphs_dir)
        _circadian_performance_curve(tests_df, results, graphs_dir)

        # Data-quality guards for self-report graphs
        sr_days = _count_self_report_days(tests_df)
        if sr_days >= 7:
            _self_report_vs_ready(tests_df, results, graphs_dir)
            _stress_sleepiness_heatmap(tests_df, results, graphs_dir)

        # Score distributions: suppress when any type has n < 20
        min_n = _min_type_count(tests_df)
        if min_n >= 20:
            _score_distributions(tests_df, results, graphs_dir)

    if metrics_df is not None:
        _hrv_night_profile(metrics_df, results, graphs_dir)

    if ss and ss.get("nights"):
        _sleep_architecture(ss, graphs_dir)
        _sleep_debt_tracker(ss, graphs_dir)
        _recovery_trend(ss, graphs_dir)

    # Strain vs recovery: suppress when strain variability is "none"
    strain_daily = strain.get("daily_strain", [])
    strain_scores = [d["strain_score"] for d in strain_daily if d.get("strain_score") is not None]
    has_strain_var = len(strain_scores) >= 3 and float(np.std(strain_scores)) > 0.5
    if has_strain_var and ss and ss.get("recovery"):
        _strain_vs_recovery(strain, ss, graphs_dir)

    plt.close("all")


def _count_self_report_days(tests_df) -> int:
    sr_cols = ["stress", "sleepiness", "sharpness"]
    available = [c for c in sr_cols if c in tests_df.columns]
    if not available:
        return 0
    return int(tests_df[available].notna().any(axis=1).sum())


def _min_type_count(tests_df) -> int:
    types = ["READY", "AGILITY", "FOCUS"]
    counts = [len(tests_df[tests_df["type"] == t]) for t in types if not tests_df[tests_df["type"] == t].empty]
    return min(counts) if counts else 0


# =========================================================================
# Merged: Ready score trajectory with Agility/Focus overlays
# (replaces separate agility_focus_trajectory.png and weekly_readiness_summary.png)
# =========================================================================

def _ready_score_with_overlays(tests_df, results, graphs_dir):
    perf = results.get("performance", {})
    daily_ready = perf.get("daily_ready", [])
    daily_ag = perf.get("daily_agility", [])
    daily_fc = perf.get("daily_focus", [])

    if not daily_ready:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime([d["date"] for d in daily_ready])
    means = [d["mean"] for d in daily_ready]
    rolling = [d.get("rolling_7d") for d in daily_ready]

    for tier, lo, hi in _TIER_BOUNDS:
        ax.axhspan(lo, hi, alpha=0.08, color=_TIER_COLORS[tier])

    ax.plot(dates, means, "o-", color="#2c3e50", markersize=5, linewidth=1.5, label="Ready (daily mean)")
    if any(r is not None for r in rolling):
        ax.plot(dates, rolling, "-", color="#e74c3c", linewidth=2.5, label="Ready 7-day avg")

    baseline = perf.get("ready_baseline")
    if baseline:
        ax.axhline(baseline, color="#27ae60", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline})")

    # Overlay Agility on secondary axis
    if daily_ag:
        ax2 = ax.twinx()
        dates_ag = pd.to_datetime([d["date"] for d in daily_ag])
        means_ag = [d["mean"] for d in daily_ag]
        ax2.bar(dates_ag, means_ag, width=0.6, color="#3498db", alpha=0.25, label="Agility")
        ax2.set_ylabel("Agility Score", color="#3498db", alpha=0.6)
        ax2.tick_params(axis="y", labelcolor="#3498db")

    ax.set_title("Ready Score Trajectory (with Agility overlay)")
    ax.set_ylabel("Ready Score")

    handles, labels = ax.get_legend_handles_labels()
    if daily_ag:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, loc="upper right", fontsize=9)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "ready_score_trajectory.png"), dpi=150)
    plt.close(fig)


def _circadian_performance_curve(tests_df, results, graphs_dir):
    circ = results.get("circadian", {})
    if tests_df is None:
        return

    ready = tests_df[tests_df["type"] == "READY"]
    if ready.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(ready)
    ax.scatter(ready["hour"], ready["score"], alpha=0.4, s=30, color="#2c3e50", label=f"Observed (n={n})")

    fitted = circ.get("fitted_curve", [])
    if fitted:
        t = [p["hour"] for p in fitted]
        y = [p["score"] for p in fitted]
        ax.plot(t, y, "-", color="#e74c3c", linewidth=2.5, label="Cosinor fit")

    peak = circ.get("estimated_peak_window")
    if peak and circ.get("acrophase_hour"):
        acro = circ["acrophase_hour"]
        ax.axvline(acro, color="#27ae60", linestyle="--", alpha=0.7, label=f"Peak ≈ {peak}")

    ax.set_title("Circadian Performance Curve (Ready by Time-of-Day)")
    ax.set_xlabel("Hour of Day (local)")
    ax.set_ylabel("Ready Score")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "circadian_performance_curve.png"), dpi=150)
    plt.close(fig)


def _self_report_vs_ready(tests_df, results, graphs_dir):
    if tests_df is None:
        return

    ready = tests_df[tests_df["type"] == "READY"].copy()
    sr_cols = ["stress", "sleepiness", "sharpness"]
    available = [c for c in sr_cols if c in ready.columns and ready[c].notna().sum() >= 3]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        sub = ready[[col, "score"]].dropna()
        ax.scatter(sub[col], sub["score"], alpha=0.5, s=40, color="#3498db")

        if len(sub) >= 5:
            r, p = stats.spearmanr(sub[col], sub["score"])
            ax.set_title(f"{col.title()} vs Ready\nSpearman r={r:.2f}, p={p:.3f}")
            z = np.polyfit(sub[col], sub["score"], 1)
            x_line = np.linspace(sub[col].min(), sub[col].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), "--", color="#e74c3c", linewidth=1.5)
        else:
            ax.set_title(f"{col.title()} vs Ready")

        ax.set_xlabel(f"{col.title()} (self-report /10)")
        ax.set_ylabel("Ready Score")

    fig.suptitle("Self-Report vs Ready Score", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "self_report_vs_ready.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _hrv_night_profile(metrics_df, results, graphs_dir):
    if metrics_df is None:
        return

    hrv = results.get("hrv", {})
    ts = hrv.get("timeseries", [])
    if not ts:
        return

    ts_df = pd.DataFrame(ts)
    if ts_df.empty:
        return

    ts_df["datetime_et"] = pd.to_datetime(ts_df["datetime_et"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(ts_df["datetime_et"], ts_df["heart_rate_mean"], color="#e74c3c", linewidth=1, alpha=0.8)
    ax1.set_ylabel("Heart Rate (BPM)")
    ax1.set_title("Night Profile: HR & HRV")
    ax1.axhline(65, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, label="HR 65 BPM threshold")
    ax1.legend(fontsize=8)

    rmssd_baseline = hrv.get("rmssd_sleep_peak")
    ax2.plot(ts_df["datetime_et"], ts_df["cardio_RMSSD_ms"], color="#3498db", linewidth=1, alpha=0.8)
    if rmssd_baseline:
        ax2.axhline(rmssd_baseline, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.7,
                     label=f"RMSSD baseline ({rmssd_baseline:.0f} ms)")
    ax2.set_ylabel("RMSSD (ms)")
    ax2.set_xlabel("Time")
    ax2.legend(fontsize=8)

    sleep_data = results.get("sleep", {})
    latest = sleep_data.get("latest_night")
    if latest:
        onset = pd.to_datetime(latest["sleep_onset"])
        offset = pd.to_datetime(latest["sleep_offset"])
        for ax in (ax1, ax2):
            ax.axvline(onset, color="#27ae60", linestyle="--", alpha=0.7, label="Sleep onset")
            ax.axvline(offset, color="#e67e22", linestyle="--", alpha=0.7, label="Sleep offset")
        ax1.legend(fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "hrv_night_profile.png"), dpi=150)
    plt.close(fig)


def _score_distributions(tests_df, results, graphs_dir):
    if tests_df is None:
        return

    perf = results.get("performance", {})
    types = ["READY", "AGILITY", "FOCUS"]
    available = [t for t in types if not tests_df[tests_df["type"] == t].empty]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, ttype in zip(axes, available):
        scores = tests_df.loc[tests_df["type"] == ttype, "score"].dropna()
        ax.hist(scores, bins=20, color="#3498db", alpha=0.7, edgecolor="#2c3e50")

        info = perf.get(ttype.lower(), {})
        if info.get("peak"):
            ax.axvline(info["peak"], color="#e74c3c", linestyle="--", label=f"Peak ({info['peak']:.0f})")
        if info.get("iqr") and info["iqr"][0] is not None:
            ax.axvspan(info["iqr"][0], info["iqr"][1], alpha=0.15, color="#f1c40f", label="IQR")

        baseline = perf.get("ready_baseline") if ttype == "READY" else None
        if baseline:
            ax.axvline(baseline, color="#27ae60", linestyle="--", label=f"Baseline ({baseline:.0f})")

        ax.set_title(f"{ttype} Distribution (n={len(scores)})")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Score Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "score_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stress_sleepiness_heatmap(tests_df, results, graphs_dir):
    if tests_df is None:
        return

    ready = tests_df[tests_df["type"] == "READY"].copy()
    if "stress" not in ready.columns or "sleepiness" not in ready.columns:
        return

    sub = ready[["stress", "sleepiness", "score"]].dropna()
    if len(sub) < 5:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    stress_bins = np.arange(0, 11, 2)
    sleep_bins = np.arange(0, 11, 2)
    sub["stress_bin"] = pd.cut(sub["stress"], bins=stress_bins, include_lowest=True)
    sub["sleep_bin"] = pd.cut(sub["sleepiness"], bins=sleep_bins, include_lowest=True)

    pivot = sub.groupby(["sleep_bin", "stress_bin"], observed=True)["score"].mean().unstack()

    if pivot.empty:
        plt.close(fig)
        return

    cmap = LinearSegmentedColormap.from_list("readiness", ["#e74c3c", "#f1c40f", "#2ecc71"])
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", origin="lower",
                   vmin=120, vmax=190)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Stress (binned)")
    ax.set_ylabel("Sleepiness (binned)")
    ax.set_title("Mean Ready Score by Stress × Sleepiness")
    fig.colorbar(im, ax=ax, label="Mean Ready Score")
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "stress_sleepiness_heatmap.png"), dpi=150)
    plt.close(fig)


# =========================================================================
# Whoop-level graphs (enhanced)
# =========================================================================


def _sleep_architecture(ss: dict, graphs_dir: str):
    nights = ss.get("nights", [])
    if not nights:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(nights) * 2), 6))

    labels = []
    for n in nights:
        tag = " (nap)" if n.get("session_type") == "nap" else ""
        labels.append(f"{n.get('night_date', '?')}{tag}")

    deep = [n.get("deep_min", 0) for n in nights]
    rem = [n.get("rem_min", 0) for n in nights]
    light = [n.get("light_min", 0) for n in nights]
    wake = [n.get("wake_min", 0) for n in nights]

    x = np.arange(len(labels))
    w = 0.5

    bar_alpha = []
    for n in nights:
        bar_alpha.append(0.4 if n.get("session_type") == "nap" else 1.0)

    ax.bar(x, deep, w, label="Deep", color="#1a5276")
    ax.bar(x, rem, w, bottom=deep, label="REM", color="#2980b9")
    ax.bar(x, light, w, bottom=[d + r for d, r in zip(deep, rem)], label="Light", color="#85c1e9")
    ax.bar(x, wake, w, bottom=[d + r + l for d, r, l in zip(deep, rem, light)], label="Wake", color="#f5b7b1")

    # Dim nap bars
    for i, n in enumerate(nights):
        if n.get("session_type") == "nap":
            for bar_container in ax.containers:
                bar_container[i].set_alpha(0.35)

    # REM target band
    max_total = max((n.get("total_sleep_min", 0) for n in nights), default=0)
    rem_target_line = max_total * 0.20 if max_total else 0
    if rem_target_line > 0:
        ax.axhline(rem_target_line, color="#2980b9", linestyle=":", alpha=0.5, label="~20% REM target")

    for i, n in enumerate(nights):
        total = n.get("total_sleep_min", 0)
        if total > 0:
            dp = n.get("deep_pct", 0)
            rp = n.get("rem_pct", 0)
            ax.text(i, total + 5, f"D:{dp}% R:{rp}%", ha="center", va="bottom", fontsize=8,
                    color="#2c3e50")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Minutes")
    ax.set_title("Sleep Architecture per Night\n(Targets: Deep 15-20%, REM 20-25%)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "sleep_architecture.png"), dpi=150)
    plt.close(fig)


def _sleep_debt_tracker(ss: dict, graphs_dir: str):
    nights = ss.get("nights", [])
    primary = [n for n in nights if n.get("session_type") != "nap"]
    if not primary:
        return

    has_debt = any(n.get("sleep_debt_min") is not None for n in primary)
    has_needed = any(n.get("sleep_needed_min") is not None for n in primary)
    if not has_debt and not has_needed:
        return

    fig, ax1 = plt.subplots(figsize=(max(8, len(primary) * 2), 6))

    labels = [n.get("night_date", "?") for n in primary]
    x = np.arange(len(labels))
    w = 0.3

    actual = [n.get("total_sleep_min", 0) / 60 for n in primary]
    needed = [n.get("sleep_needed_min", 0) / 60 if n.get("sleep_needed_min") else 0 for n in primary]

    ax1.bar(x - w / 2, actual, w, label="Actual sleep (h)", color="#2ecc71", alpha=0.8)
    ax1.bar(x + w / 2, needed, w, label="Sleep needed (h)", color="#95a5a6", alpha=0.6)
    ax1.set_ylabel("Hours")
    ax1.set_title("Sleep Debt Tracker")

    if has_debt:
        ax2 = ax1.twinx()
        debts = [n.get("sleep_debt_min", 0) / 60 if n.get("sleep_debt_min") is not None else 0 for n in primary]
        ax2.plot(x, debts, "o-", color="#e74c3c", linewidth=2.5, markersize=7, label="Sleep debt (h)")
        ax2.axhline(0, color="#e74c3c", linestyle=":", alpha=0.4)
        ax2.set_ylabel("Sleep Debt (hours)", color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")

        # 3-day projected debt line
        if len(debts) >= 2:
            slope = debts[-1] - debts[-2]
            projected = [debts[-1] + slope * (i + 1) for i in range(3)]
            proj_x = [x[-1] + i + 1 for i in range(3)]
            ax2.plot(proj_x, projected, "--", color="#e74c3c", alpha=0.4, linewidth=1.5, label="3-day projection")

        # Recommended bedtime annotation from sleep_debt analysis
        debt_info = ss.get("sleep_debt")
        if debt_info and debt_info.get("current_debt_hours"):
            debt_h = debt_info["current_debt_hours"]
            ax1.annotate(f"Debt: {debt_h}h", xy=(x[-1], actual[-1]),
                         xytext=(x[-1] - 0.5, max(actual) + 0.5),
                         fontsize=9, color="#e74c3c", fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color="#e74c3c", alpha=0.6))

        lines2, labels2 = ax2.get_legend_handles_labels()
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    else:
        ax1.legend(loc="upper left", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "sleep_debt_tracker.png"), dpi=150)
    plt.close(fig)


def _recovery_trend(ss: dict, graphs_dir: str):
    recovery = ss.get("recovery")
    if not recovery or not recovery.get("history"):
        return

    nights = ss.get("nights", [])
    primary = [n for n in nights if n.get("session_type") != "nap"]
    scores = recovery["history"]
    labels = [n.get("night_date", f"Session {i+1}") for i, n in enumerate(primary)]
    if len(labels) < len(scores):
        labels += [f"Session {i+1}" for i in range(len(labels), len(scores))]

    fig, ax = plt.subplots(figsize=(max(8, len(scores) * 2), 5))

    ax.axhspan(67, 100, alpha=0.15, color="#2ecc71", label="Green zone (67-100%)")
    ax.axhspan(34, 67, alpha=0.15, color="#f1c40f", label="Yellow zone (34-66%)")
    ax.axhspan(0, 34, alpha=0.15, color="#e74c3c", label="Red zone (0-33%)")

    x = np.arange(len(scores))
    colors = ["#2ecc71" if s >= 67 else "#f1c40f" if s >= 34 else "#e74c3c" for s in scores]
    ax.bar(x, scores, color=colors, alpha=0.8, edgecolor="#2c3e50")

    mean = recovery.get("personal_mean")
    if mean:
        ax.axhline(mean, color="#8e44ad", linestyle="--", linewidth=1.5, label=f"Personal avg ({mean}%)")

    dsg = recovery.get("days_since_green_zone")
    subtitle = f"Days since green: {dsg}" if dsg is not None else ""
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(scores)], rotation=45, ha="right")
    ax.set_ylabel("Recovery Score (%)")
    ax.set_title(f"Recovery Trend\n{subtitle}" if subtitle else "Recovery Trend")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "recovery_trend.png"), dpi=150)
    plt.close(fig)


def _strain_vs_recovery(strain: dict, ss: dict, graphs_dir: str):
    daily_strain = strain.get("daily_strain", [])
    nights = ss.get("nights", [])
    if not daily_strain or not nights:
        return

    recovery_by_date = {n.get("night_date"): n.get("recovery_score") for n in nights
                        if n.get("recovery_score") is not None and n.get("session_type") != "nap"}

    points = []
    for s in daily_strain:
        strain_date = s["date"]
        rec = recovery_by_date.get(strain_date)
        if rec is not None:
            points.append((s["strain_score"], rec))

    if len(points) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    colors = ["#2ecc71" if y >= 67 else "#f1c40f" if y >= 34 else "#e74c3c" for y in ys]

    ax.scatter(xs, ys, c=colors, s=80, edgecolors="#2c3e50", linewidth=1, zorder=5)

    if len(points) >= 3:
        z = np.polyfit(xs, ys, 1)
        x_fit = np.linspace(min(xs), max(xs), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), "--", color="#8e44ad", linewidth=1.5, label=f"Trend (slope={z[0]:.1f})")

    ax.set_xlabel("Strain Score (0-21)")
    ax.set_ylabel("Recovery Score (%)")
    ax.set_title("Strain vs Recovery")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(graphs_dir, "strain_vs_recovery.png"), dpi=150)
    plt.close(fig)
