# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40",
#   "python-dotenv>=1.0",
# ]
# ///
"""One-shot smoke test for pgx-digest's Claude wiring.

Loads ANTHROPIC_API_KEY from .env, calls Claude Haiku 4.5, prints the
response. Does not print the API key.

Run:
    uv run smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).parent / ".env")

key = os.environ.get("ANTHROPIC_API_KEY", "")
if not key:
    sys.stderr.write("ERROR: ANTHROPIC_API_KEY not set in .env\n")
    sys.exit(1)

print(f"Key loaded from .env (length={len(key)} chars).")

client = anthropic.Anthropic()

print("Calling claude-haiku-4-5...")
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=128,
    messages=[
        {
            "role": "user",
            "content": (
                "Respond with exactly the phrase "
                "'pgx-digest wiring works' and nothing else."
            ),
        }
    ],
)

text = next(b.text for b in response.content if b.type == "text")
print(f"\nResponse:    {text}")
print(f"Model:       {response.model}")
print(f"Stop reason: {response.stop_reason}")
print(
    f"Usage:       input={response.usage.input_tokens} "
    f"output={response.usage.output_tokens}"
)
print("\nSmoke test passed.")
