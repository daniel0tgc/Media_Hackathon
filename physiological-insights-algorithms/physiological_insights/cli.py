"""CLI entry point for the physiological insights pipeline."""

import argparse
import json
import os

from physiological_insights.ingest import load_test_results, load_decoded_metrics, load_sleep_sessions
from physiological_insights.self_report import parse_all_comments
from physiological_insights.performance import analyse_performance
from physiological_insights.hrv import analyse_hrv
from physiological_insights.sleep import analyse_sleep
from physiological_insights.sleep_sessions import analyse_sleep_sessions
from physiological_insights.activity import analyse_activity
from physiological_insights.strain import analyse_strain
from physiological_insights.circadian import analyse_circadian
from physiological_insights.readiness import assign_readiness_tiers
from physiological_insights.patterns import detect_patterns
from physiological_insights.visualizations import generate_all_graphs
from physiological_insights.context_packet import build_context_packet
from physiological_insights.agent_payload import build_agent_payload


def main():
    parser = argparse.ArgumentParser(
        prog="physiological_insights",
        description="Research-backed physiological data analysis pipeline.",
    )
    parser.add_argument("--test-csv", help="Path to test results CSV (READY/AGILITY/FOCUS scores)")
    parser.add_argument("--sleep-csv", help="Path to sleep sessions CSV (sleep stages, recovery, debt)")
    parser.add_argument("--metrics-csv", help="Path to decoded metrics CSV (sensor epoch data)")
    parser.add_argument("--user-name", required=True, help="User name for per-user output folder")
    parser.add_argument("--output", default=None, help="Full analysis JSON path (default: output/{user}/analysis_full.json)")
    parser.add_argument("--graphs-dir", default=None, help="Directory for graph PNGs (default: output/{user}/graphs)")
    parser.add_argument("--briefing", default=None, help="Path for condensed agent briefing JSON (triggers Tier 2 LLM)")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()

    if not args.test_csv and not args.metrics_csv and not args.sleep_csv:
        parser.error("At least one of --test-csv, --sleep-csv, or --metrics-csv is required.")

    user_dir = os.path.join("output", args.user_name)
    full_path = args.output or os.path.join(user_dir, "analysis_full.json")
    payload_path = os.path.join(os.path.dirname(full_path), "agent_payload.json")
    graphs_dir = args.graphs_dir or os.path.join(user_dir, "graphs")

    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # --- Tier 1: Deterministic pipeline ---
    print(f"[Tier 1] Pipeline for user: {args.user_name}")

    print("[Tier 1] Loading data...")
    tests_df = load_test_results(args.test_csv) if args.test_csv else None
    sleep_df = load_sleep_sessions(args.sleep_csv) if args.sleep_csv else None
    metrics_df = load_decoded_metrics(args.metrics_csv) if args.metrics_csv else None

    results = {}

    if tests_df is not None:
        print("[Tier 1] Parsing self-reports from comments...")
        tests_df = parse_all_comments(tests_df)

        print("[Tier 1] Analysing performance scores...")
        results["performance"] = analyse_performance(tests_df)

        print("[Tier 1] Fitting circadian model...")
        results["circadian"] = analyse_circadian(tests_df, sleep_df=sleep_df)

    if sleep_df is not None:
        print("[Tier 1] Analysing sleep sessions (Whoop-level)...")
        results["sleep_sessions"] = analyse_sleep_sessions(sleep_df)

    if metrics_df is not None:
        print("[Tier 1] Analysing HRV...")
        results["hrv"] = analyse_hrv(metrics_df)

        print("[Tier 1] Detecting sleep from sensors...")
        results["sleep"] = analyse_sleep(metrics_df)

        print("[Tier 1] Classifying activity...")
        results["activity"] = analyse_activity(metrics_df)

        print("[Tier 1] Computing strain scores...")
        results["strain"] = analyse_strain(metrics_df)

    if tests_df is not None:
        print("[Tier 1] Assigning readiness tiers...")
        results["readiness"] = assign_readiness_tiers(results)

    print("[Tier 1] Detecting multi-day patterns...")
    results["patterns"] = detect_patterns(tests_df, results)

    print("[Tier 1] Assembling full analysis packet...")
    packet = build_context_packet(tests_df, metrics_df, sleep_df, results, graphs_dir)

    with open(full_path, "w") as f:
        json.dump(packet, f, indent=2, default=str)
    print(f"[Tier 1] analysis_full.json -> {full_path}")

    print("[Tier 1] Building agent payload...")
    payload = build_agent_payload(packet, results)

    with open(payload_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    payload_tokens = len(json.dumps(payload, default=str)) // 4
    print(f"[Tier 1] agent_payload.json -> {payload_path}  (~{payload_tokens} tokens)")

    print("[Tier 1] Generating graphs...")
    generate_all_graphs(tests_df, metrics_df, sleep_df, results, packet, graphs_dir)
    print(f"[Tier 1] Graphs written to {graphs_dir}/")

    # --- Tier 2: Analyst LLM (optional) ---
    if args.briefing:
        briefing_path = args.briefing
        os.makedirs(os.path.dirname(briefing_path) or ".", exist_ok=True)
        print("[Tier 2] Running analyst LLM...")
        from physiological_insights.analyst import generate_briefing
        briefing = generate_briefing(packet, provider=args.llm_provider, model=args.llm_model)
        with open(briefing_path, "w") as f:
            json.dump(briefing, f, indent=2, default=str)
        print(f"[Tier 2] Agent briefing written to {briefing_path}")

    print("Done.")
