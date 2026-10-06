"""
Build the demo video, end to end, from things in this repository.

    export OPENROUTER_API_KEY=...          # narration only
    python demo/film.py narrate            # demo/out/audio/<scene>.mp3
    python demo/film.py slides             # demo/out/slides/<scene>.png
    python demo/film.py record             # demo/out/live/ - run while the arena is live
    python demo/film.py assemble           # demo/out/nullpointerexception-demo.mp4

For `record`, start a live run first, e.g.

    ARENA_SEED=33 SCENARIO=graded ARENA_ROUND_SECONDS=1.0 ARENA_LEASE_SECONDS=6 \
    ARENA_START_DELAY=10 docker compose --profile agent up -d

Needs ffmpeg, and Playwright with a Chromium (CHROMIUM=/path/to/chrome to
choose one). Nothing here is part of the agent.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SCRIPT = json.loads((HERE / "script.json").read_text())
W, H, FPS = 1920, 1080, 30
PAD_IN, PAD_OUT = 0.5, 0.9          # seconds of picture before and after each line


def chromium() -> Dict[str, str]:
    path = os.environ.get("CHROMIUM")
    if not path:
        hits = sorted(glob.glob(os.path.expanduser(
            "~/Library/Caches/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium")))
        path = hits[-1] if hits else None
    return {"executable_path": path} if path else {}


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


# ---------------------------------------------------------------------------
def narrate(force: bool = False) -> None:
    sys.path.insert(0, str(HERE))
    from tts import speak
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    for s in SCRIPT["scenes"]:
        path = OUT / "audio" / f"{s['id']}.mp3"
        if path.exists() and not force:
            continue
        raw = path.with_suffix(".raw.mp3")
        speak(s["text"], raw, voice=SCRIPT.get("voice", "flux-marcus-en"),
              model=SCRIPT.get("model", "deepgram/flux-tts"))
        # A touch faster, pitch preserved: the provider has no speed control.
        ff("-i", str(raw), "-filter:a", f"atempo={float(SCRIPT.get('tempo', 1.0)):.3f}",
           "-b:a", "192k", str(path))
        raw.unlink()
        print(f"{s['id']:<10} {duration(path):5.1f}s")


def slides() -> None:
    from playwright.sync_api import sync_playwright
    (OUT / "slides").mkdir(parents=True, exist_ok=True)
    url = (HERE / "slides.html").as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch(**chromium())
        page = browser.new_page(viewport={"width": W, "height": H})
        for s in SCRIPT["scenes"]:
            if s["visual"] != "slide":
                continue
            page.goto(f"{url}#{s['id']}")
            page.wait_for_timeout(400)
            page.screenshot(path=str(OUT / "slides" / f"{s['id']}.png"))
            print("slide", s["id"])
        browser.close()


def record(url: str = "http://localhost:8090", tail: float = 6.0, cap: float = 300.0) -> None:
    """Screenshot the live dashboard ~10 times a second until the run ends (or `cap` s)."""
    from playwright.sync_api import sync_playwright
    d = OUT / "live"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()
    stamps: List[float] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(**chromium())
        page = browser.new_page(viewport={"width": W, "height": H})
        page.goto(url)
        page.wait_for_function("document.getElementById('waiting').classList.contains('hidden')",
                               timeout=180_000)
        t0, ended = time.monotonic(), None
        while True:
            now = time.monotonic()
            page.screenshot(path=str(d / f"{len(stamps):05d}.png"))
            stamps.append(now - t0)
            final = page.evaluate("document.getElementById('round').textContent.startsWith('FINAL')")
            if final and ended is None:
                ended = now
            if (ended is not None and now - ended > tail) or now - t0 > cap:
                break
            time.sleep(max(0.0, 0.1 - (time.monotonic() - now)))
        browser.close()
    (d / "stamps.json").write_text(json.dumps(stamps))
    print(f"{len(stamps)} frames over {stamps[-1]:.1f}s")


def capture(url: str = "http://localhost:8090/api/state",
            out: Path = HERE.parent / "site" / "data" / "live-run.json",
            label: str = "", cap: float = 400.0) -> None:
    """Save a live run's telemetry for the website's in-browser replay.

    One frame per round (this round's decision, the leaderboard, the status);
    the per-round history is stored once and cut to each frame's round by the
    dashboard, which keeps the file small.
    """
    import httpx
    frames: List[dict] = []
    last_round, final, t0 = None, None, time.monotonic()
    while time.monotonic() - t0 < cap:
        try:
            s = httpx.get(url, timeout=2.0).json()
        except (httpx.HTTPError, ValueError):
            time.sleep(0.5)
            continue
        d = s.get("decision") or {}
        r = d.get("round")
        done = bool((s.get("status") or {}).get("finished"))
        if r and r != last_round and not done:
            frames.append({"decision": d, "leaderboard": s.get("leaderboard"),
                           "status": s.get("status")})
            last_round = r
        if done:
            final = s
            break
        time.sleep(0.15)
    if final is None:
        raise SystemExit("the run did not finish within the cap")
    frames.append({"decision": {"round": (final["status"].get("total_rounds") or 0) + 1,
                                "final": True},
                   "leaderboard": final.get("leaderboard"), "status": final.get("status")})
    data = {"meta": {"label": label}, "team": final.get("team"),
            "rounds": final.get("rounds"), "decisions": final.get("decisions"), "frames": frames}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), default=float))
    print(f"{len(frames)} frames, {out.stat().st_size // 1024} KB -> {out}")


# ---------------------------------------------------------------------------
def _segment_slide(scene: dict, audio: Path, out: Path) -> None:
    length = PAD_IN + duration(audio) + PAD_OUT
    img = OUT / "slides" / f"{scene['id']}.png"
    # A still slide, held perfectly still. (A slow zoom via zoompan was tried
    # and dropped: it snaps each frame to whole pixels, so text visibly wobbles.)
    vf = (f"scale={W}:{H},fps={FPS},"
          f"fade=t=in:st=0:d=0.35,fade=t=out:st={length - 0.4:.2f}:d=0.4,format=yuv420p")
    ff("-loop", "1", "-framerate", str(FPS), "-i", str(img), "-i", str(audio),
       "-filter_complex", f"[0:v]{vf}[v];[1:a]adelay={int(PAD_IN * 1000)}|{int(PAD_IN * 1000)},"
                          f"apad=whole_dur={length:.3f}[a]",
       "-map", "[v]", "-map", "[a]", "-t", f"{length:.3f}", "-r", str(FPS),
       "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
       "-ar", "48000", "-ac", "2", str(out))


def _segment_live(scene: dict, audio: Path, out: Path) -> None:
    d = OUT / "live"
    stamps = json.loads((d / "stamps.json").read_text())
    real = OUT / "live.mp4"
    lines = []
    for i, t in enumerate(stamps):
        nxt = stamps[i + 1] if i + 1 < len(stamps) else t + 0.1
        lines += [f"file '{d / f'{i:05d}.png'}'", f"duration {nxt - t:.4f}"]
    lines.append(f"file '{d / f'{len(stamps) - 1:05d}.png'}'")
    (d / "frames.txt").write_text("\n".join(lines))
    ff("-f", "concat", "-safe", "0", "-i", str(d / "frames.txt"), "-vf",
       f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-crf", "16", "-preset", "slow", str(real))

    length = PAD_IN + duration(audio) + PAD_OUT
    speed = duration(real) / length
    label = f"LIVE ARENA  ·  graded scenario, injected faults  ·  1 s rounds, shown at {speed:.1f}x"
    vf = (f"setpts=PTS/{speed:.4f},fps={FPS},"
          f"drawbox=x=170:y=ih-58:w=1000:h=42:color=0x070b16@0.9:t=fill,"
          f"drawtext=text='{label}':x=186:y=h-47:fontsize=21:fontcolor=0x00a6c1:"
          f"fontfile=/System/Library/Fonts/Supplemental/Arial Bold.ttf,"
          f"fade=t=in:st=0:d=0.35,fade=t=out:st={length - 0.4:.2f}:d=0.4,format=yuv420p")
    ff("-i", str(real), "-i", str(audio),
       "-filter_complex", f"[0:v]{vf}[v];[1:a]adelay={int(PAD_IN * 1000)}|{int(PAD_IN * 1000)},"
                          f"apad=whole_dur={length:.3f}[a]",
       "-map", "[v]", "-map", "[a]", "-t", f"{length:.3f}",
       "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
       "-ar", "48000", "-ac", "2", str(out))


def assemble() -> Path:
    seg_dir = OUT / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in SCRIPT["scenes"]:
        audio = OUT / "audio" / f"{s['id']}.mp3"
        seg = seg_dir / f"{len(parts):02d}-{s['id']}.mp4"
        (_segment_live if s["visual"] == "live" else _segment_slide)(s, audio, seg)
        parts.append(seg)
        print(f"{s['id']:<10} {duration(seg):5.1f}s")
    listing = seg_dir / "list.txt"
    listing.write_text("\n".join(f"file '{p}'" for p in parts))
    joined = seg_dir / "joined.mp4"
    ff("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined))
    # Normalise loudness to the usual web/streaming level.
    final = OUT / "nullpointerexception-demo.mp4"
    ff("-i", str(joined), "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
       "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(final))
    print(f"{final}  {duration(final):.1f}s")
    return final


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["narrate", "slides", "record", "capture", "assemble"])
    p.add_argument("--label", default="", help="capture: caption shown on the replay")
    p.add_argument("--force", action="store_true", help="re-narrate cached scenes")
    p.add_argument("--env-file", default=None,
                   help="read KEY=VALUE lines (e.g. OPENROUTER_API_KEY) from this file")
    a = p.parse_args()
    if a.env_file:
        for line in Path(a.env_file).read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key, value)
    {"narrate": lambda: narrate(a.force), "slides": slides, "record": record,
     "capture": lambda: capture(label=a.label), "assemble": assemble}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
