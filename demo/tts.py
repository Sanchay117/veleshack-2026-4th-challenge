"""
Narration for the demo video: text in, MP3 out, via OpenRouter's speech API.

    OPENROUTER_API_KEY=... python demo/tts.py "Hello swarm" out.mp3 [--voice flux-marcus-en]

The key is read from the environment only and never written anywhere.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/audio/speech"
MODEL = "deepgram/flux-tts"
VOICE = "flux-marcus-en"


def speak(text: str, out: Path, voice: str = VOICE, model: str = MODEL,
          speed: float = 1.0, attempts: int = 4) -> Path:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")
    body = {"model": model, "input": text, "voice": voice, "response_format": "mp3"}
    if speed != 1.0:
        body["speed"] = speed   # not every provider accepts it (Deepgram Flux does not)
    for attempt in range(attempts):
        r = httpx.post(URL, json=body, timeout=120,
                       headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 200 and r.content:
            out.write_bytes(r.content)
            return out
        if r.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
            time.sleep(2 ** attempt)
            continue
        raise SystemExit(f"TTS failed: HTTP {r.status_code}: {r.text[:300]}")
    raise SystemExit("TTS failed")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("out")
    p.add_argument("--voice", default=VOICE)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--speed", type=float, default=1.0)
    a = p.parse_args()
    speak(a.text, Path(a.out), a.voice, a.model, a.speed)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
