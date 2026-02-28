"""Main agent loop: watches iMessage via imsg CLI, responds via Claude."""

import json
import os
import subprocess
import sys
import signal

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hackathon.database import init_db
from hackathon.brain import get_response

# Target phone number (from .env or hardcoded for hackathon)
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")
USER_PHONE = os.environ.get("USER_PHONE", "+18455320691")
# Daniel's number — only respond to this number in live mode
DANIEL_PHONE = "+12135689314"


def send_imessage(phone: str, text: str):
    """Send an iMessage via imsg CLI."""
    try:
        subprocess.run(
            ["imsg", "send", "--to", phone, "--text", text],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  -> Sent to {phone}: {text[:80]}...")
    except FileNotFoundError:
        print("  ! imsg not found. Install: brew install steipete/tap/imsg")
        print(f"  -> Would send to {phone}: {text}")
    except subprocess.CalledProcessError as e:
        print(f"  ! imsg send failed: {e.stderr}")


def watch_and_respond():
    """Watch for incoming iMessages and respond."""
    print("Nervous System Agent starting...")
    print(f"   Watching for messages from: {USER_PHONE}")
    print(f"   Press Ctrl+C to stop\n")

    # Start imsg watch as a streaming subprocess
    try:
        proc = subprocess.Popen(
            ["imsg", "watch", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )
    except FileNotFoundError:
        print("imsg not found. Install: brew install steipete/tap/imsg")
        print("   Falling back to interactive mode...\n")
        interactive_mode()
        return

    def cleanup(sig, frame):
        print("\n\nShutting down...")
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    print("Watching for incoming iMessages...\n")

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        # CRITICAL: Skip messages FROM us (prevents infinite loop)
        if msg.get("is_from_me", False):
            continue

        sender = msg.get("sender", "")
        text = msg.get("text", "")

        if not text:
            continue

        # Only respond to Daniel's messages during testing
        normalized_sender = sender.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
        if not normalized_sender.startswith("+"):
            normalized_sender = "+1" + normalized_sender if len(normalized_sender) == 10 else "+" + normalized_sender
        if normalized_sender != DANIEL_PHONE:
            continue

        print(f"From {sender}: {text}")

        # Normalize phone for DB
        phone = sender.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
        if not phone.startswith("+"):
            phone = "+1" + phone if len(phone) == 10 else "+" + phone

        try:
            response = get_response(phone, text)
            print(f"Response: {response}")
            send_imessage(sender, response)
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()


def interactive_mode():
    """Fallback: interactive terminal mode for testing without imsg."""
    print("Interactive Mode (type messages, Ctrl+C to quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            response = get_response(USER_PHONE, user_input)
            print(f"\nAgent: {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.")
            break


if __name__ == "__main__":
    init_db()

    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive_mode()
    else:
        watch_and_respond()
