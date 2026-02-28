"""Integration test: verify the agent can process a message end-to-end."""

import os
import sys

# Add parent dir to path so 'hackathon' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hackathon.database import init_db, get_conversation_history, get_or_create_profile
from hackathon.brain import get_response


def test_first_message():
    """Test the first-breath moment -- user's very first interaction."""
    phone = "+18455320691"

    print("=" * 60)
    print("TEST: First-Breath Moment (first ever message)")
    print("=" * 60)

    response = get_response(phone, "Hey, I just got my Pison band. What is this?")
    print(f"\nUser: Hey, I just got my Pison band. What is this?")
    print(f"\nAgent: {response}")
    print()

    # Verify message was saved
    history = get_conversation_history(phone)
    assert len(history) >= 2, f"Expected at least 2 messages, got {len(history)}"
    print(f"OK - Messages saved to DB ({len(history)} total)")

    # Verify profile was created
    profile = get_or_create_profile(phone)
    assert profile is not None
    print(f"OK - User profile created")

    return response


def test_followup():
    """Test a follow-up conversation message."""
    phone = "+18455320691"

    print("\n" + "=" * 60)
    print("TEST: Follow-up message")
    print("=" * 60)

    response = get_response(phone, "I'm a college student and I want to track how my brain performs during exam season")
    print(f"\nUser: I'm a college student and I want to track how my brain performs during exam season")
    print(f"\nAgent: {response}")
    print()

    history = get_conversation_history(phone)
    print(f"OK - Conversation now has {len(history)} messages")

    return response


def test_stress_message():
    """Test how agent handles an emotional/stress message."""
    phone = "+18455320691"

    print("\n" + "=" * 60)
    print("TEST: Stress/emotional message")
    print("=" * 60)

    response = get_response(phone, "I'm so stressed about my midterm tomorrow. Haven't slept well.")
    print(f"\nUser: I'm so stressed about my midterm tomorrow. Haven't slept well.")
    print(f"\nAgent: {response}")
    print()

    return response


if __name__ == "__main__":
    print("\nNervous System Agent -- Integration Test\n")

    # Re-init DB fresh for test
    init_db()

    try:
        test_first_message()
        test_followup()
        test_stress_message()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
