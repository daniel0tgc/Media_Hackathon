# Physiological Insights Algorithms

Pipeline for turning unstructured wearable CSV exports into structured physiological intelligence:

- Standardized `analysis_output.json` context packets
- Interpretable graph artifacts (`.png`)
- Feature extraction designed for downstream AI agents

The project is built to support **any user with wearable physiological data**, not just a single person or single vendor export format.

## Product Goal

This software is the transformation layer between messy raw biometric exports and a consumer-facing AI experience.

It is designed to:

- Ingest heterogeneous CSV files from wearable workflows
- Normalize and analyze key signals (readiness, HRV, sleep, strain, circadian patterns, self-report context)
- Produce structured outputs that can be embedded in an AI agent's context
- Enable decision support against real-world constraints like calendar load, meetings, workouts, and focus blocks

In short: **raw physiology -> structured insights -> AI-ready context -> personalized recommendations.**

## What It Produces

For each user run, the pipeline outputs:

- `output/<user_name>/analysis_output.json`
- `output/<user_name>/graphs/*.png`

Current graph set includes:

- `ready_score_trajectory.png`
- `agility_focus_trajectory.png`
- `circadian_performance_curve.png`
- `self_report_vs_ready.png`
- `hrv_night_profile.png`
- `weekly_readiness_summary.png`
- `score_distributions.png`
- `stress_sleepiness_heatmap.png`
- `sleep_architecture.png`
- `sleep_debt_tracker.png`
- `recovery_trend.png`
- `strain_vs_recovery.png`

## Repository Layout

`physiological_insights/` is the core package:

- `cli.py` - CLI orchestrator and end-to-end pipeline execution
- `ingest.py` - CSV loading, validation, timezone normalization
- `self_report.py` - parsing stress/sleepiness/sharpness from comments
- `performance.py` - Ready/Agility/Focus stats and weekly summaries
- `circadian.py` - cosinor fit and time-of-day performance windows
- `hrv.py` - RMSSD baseline, trend, and HRV-derived metrics
- `sleep.py` - sensor-derived sleep onset/offset and fragmentation signals
- `sleep_sessions.py` - sleep architecture, debt, recovery, and sleep performance
- `activity.py` - physical load classification from wear epochs
- `strain.py` - daily strain scoring (Whoop-inspired 0-21 scale)
- `readiness.py` - readiness tiering and task suitability matrix
- `patterns.py` - multi-day pattern/risk detection
- `context_packet.py` - final JSON schema assembly and insight strings
- `visualizations.py` - graph rendering
- `analyst.py` - optional Tier-2 LLM summarization

Data directories in this repo currently include `Jerry_data/` and `Daniel_data/` for local runs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (for LLM briefing generation):

- Set provider API key (for example `OPENAI_API_KEY`) in your environment

## Running the Pipeline

The CLI accepts one or more data sources:

- `--test-csv` (READY/AGILITY/FOCUS test data)
- `--sleep-csv` (sleep sessions / recovery / debt data)
- `--metrics-csv` (decoded sensor epochs)

`--user-name` is required and controls output folder naming.

### Full run (all inputs)

```bash
python -m physiological_insights \
  --user-name Daniel \
  --test-csv "Daniel_data/bq-daniel-reaction-results.csv" \
  --sleep-csv "Daniel_data/bq-daniel-fatigue-results.csv" \
  --metrics-csv "Daniel_data/daniel_decoded_metrics.csv"
```

### Partial run (tests + metrics only)

```bash
python -m physiological_insights \
  --user-name Jerry \
  --test-csv "Jerry_data/bq-jerry-reaction-results-20260227-220836-1772230120565.csv" \
  --metrics-csv "Jerry_data/jerry_decoded_metrics.csv"
```

### With optional LLM briefing output

```bash
python -m physiological_insights \
  --user-name Daniel \
  --test-csv "Daniel_data/bq-daniel-reaction-results.csv" \
  --sleep-csv "Daniel_data/bq-daniel-fatigue-results.csv" \
  --metrics-csv "Daniel_data/daniel_decoded_metrics.csv" \
  --briefing "output/Daniel/agent_briefing.json" \
  --llm-provider openai \
  --llm-model gpt-4o-mini
```

## JSON Output: AI-Agent Context Contract

`analysis_output.json` is the core machine-readable contract for downstream AI systems.

Top-level sections include:

- `meta` - run timestamp, data coverage windows, user id, graph path
- `baseline` - personal baseline anchors (Ready, Agility peak, HRV peak)
- `latest_day` - most recent day summary (readiness tier, self-report, strain, activity)
- `daily_summaries` and `weekly_summaries` - trend-ready aggregates
- `trends` and `patterns_detected` - actionable directional signals
- `circadian_profile` - estimated peak/worst windows for cognitive/physical tasks
- `task_matching` - suitability map for task categories
- `sleep_sessions`, `sleep_debt`, `recovery`, `strain` - recovery and load intelligence
- `insights` - plain-language insight strings for UI/agent prompts
- `graphs` - generated artifact manifest

This schema is intended to be consumed by agents that also know calendar context, for example:

- prioritize deep work during peak readiness windows
- suggest lower-cognitive-load slots during reduced recovery
- adapt workout intensity to strain/recovery state
- recommend sleep timing shifts when circadian misalignment emerges

## Data Expectations

The pipeline is robust to sparse input, but quality improves when all three data sources are provided.

High-level expectations:

- Test CSV: contains `type`, `score`, `created_at` (+ optional comment/self-report text)
- Sleep CSV: contains sleep-stage and recovery-related fields (session-level)
- Metrics CSV: contains epoch-level wearable streams (HR, HRV proxies, motion, wear state)

If only one input type is provided, unavailable modules are skipped and output is partially populated.

## Design Principles

- **Generalizable:** not tied to a single user
- **Interpretable:** clear metrics and visualizations, not opaque scores only
- **Composable:** structured output for AI-agent pipelines
- **Incremental:** supports deterministic Tier-1 analytics with optional Tier-2 LLM compression
- **Consumer-oriented:** optimized for actionable recommendations, not just analytics dumps

## Current Limitations and Notes

- Data exports vary by source and may include outliers, duplicated exports, or mixed units.
- Some modules rely on heuristics (thresholds, rolling windows, zone boundaries) and should be calibrated as datasets expand.
- AI-agent integration with calendar/agenda is currently a downstream consumer concern; this repo prepares the context packet for that layer.
- No formal test suite is included yet.

## Next Product-Step Recommendations

- Stabilize a versioned JSON schema for agent consumers (`schema_version` + migration notes)
- Add strict data validation and outlier-handling policies at ingest
- Add cross-device normalization layer for portability beyond one vendor export style
- Add confidence scoring per insight so agents can reason about uncertainty
- Add integration adapter that merges `analysis_output.json` with calendar/workload context into a single agent payload

## License

No license file is currently present in this repository. Add one before distribution.
