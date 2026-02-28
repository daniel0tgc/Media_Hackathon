# 🧠 Nervous System Agent

**"What if your body could text you?"**

AI agent that intakes biometrics from any health wearable + user calendar + user goals → delivers cognitive/health insights via iMessage.

Built at the E14 Beyond the Lab Hackathon (MIT Media Lab, Feb 2026).

## Architecture

```
imsg watch --json
  → Python parses incoming message
  → Query SQLite (conversation history + user profile)
  → Load biometric insights (JSON from data pipeline)
  → Load calendar context (Google Calendar JSON)
  → Call Claude API with full context + system prompt (🧠 voice)
  → Store both messages in DB
  → imsg send response
  → User receives on iPhone
```

## Setup

```bash
# Create venv and install dependencies
python -m venv hackathon/venv
source hackathon/venv/bin/activate
pip install -r hackathon/requirements.txt

# Create .env with your API key
echo 'ANTHROPIC_API_KEY=your-key-here' > hackathon/.env
echo 'USER_PHONE=+1XXXXXXXXXX' >> hackathon/.env

# Install imsg CLI (macOS only)
brew install steipete/tap/imsg

# Run in interactive mode (no iMessage needed)
python hackathon/agent.py -i

# Run in live iMessage mode
python hackathon/agent.py
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| AI | Claude API (Sonnet 4) via `anthropic` SDK |
| Database | SQLite |
| Messaging | `imsg` CLI (iMessage via Mac relay) |
| Data | Pison wearable biometrics + Google Calendar |

## Team

- Jerry Chen
- Daniel TGC
