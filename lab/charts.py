"""
Render the write-up's figures as static SVG, from docs/results/results.json
and one replayed run.

    python lab/charts.py            # writes docs/img/*.svg

Palette: the four series colours are validated for the dark surface used here
(OKLCH lightness band, colour-blind separation); every series is also
direct-labelled, so identity never rests on colour alone.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sim

IMG = sim.ROOT / "docs" / "img"
RESULTS = sim.ROOT / "docs" / "results" / "results.json"

SURFACE = "#0c1324"
INK = "#e6ecff"
INK2 = "#8a97b8"
MUTED = "#56627f"
GRID = "#1d2742"
US = "#00a6c1"
COLORS = {"bot-naive-max": "#c88100", "bot-even-split": "#9f78e9",
          "bot-proportional": "#f04e7e"}
NEUTRAL = "#4a5677"
FONT = "-apple-system,BlinkMacSystemFont,Inter,Segoe UI,Roboto,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


class Svg:
    def __init__(self, w: int, h: int, title: str, subtitle: str = "") -> None:
        self.w, self.h = w, h
        self.parts: List[str] = [
            f'<rect width="{w}" height="{h}" rx="14" fill="{SURFACE}"/>',
            self._text(28, 38, title, 19, INK, weight=700),
        ]
        if subtitle:
            self.parts.append(self._text(28, 62, subtitle, 13, INK2))

    @staticmethod
    def _text(x, y, s, size=12, fill=INK2, anchor="start", weight=400, mono=False) -> str:
        fam = MONO if mono else FONT
        return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
                f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(s))}</text>')

    def text(self, *a, **k) -> None:
        self.parts.append(self._text(*a, **k))

    def add(self, s: str) -> None:
        self.parts.append(s)

    def save(self, name: str) -> Path:
        IMG.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.parts)
        path = IMG / name
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                        f'width="{self.w}" height="{self.h}">\n{body}\n</svg>\n', encoding="utf-8")
        return path


def polyline(points: Sequence[Tuple[float, float]], color: str, width: float = 2.0,
             dash: str = "") -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round"{d}/>')


def short(team: str, us: str) -> str:
    return "us" if team == us else team.replace("bot-", "")


# ---------------------------------------------------------------------------
# 1. The ladder
# ---------------------------------------------------------------------------
LABELS = {
    "template": "template (naive-max)",
    "even-split": "even-split bot",
    "proportional": "proportional bot",
    "best-response": "exact best response, no battery",
    "taper": "battery taper + floors (strong heuristic)",
    "ours-no-model": "ours, without the opponent model",
    "ours-no-plan": "ours, without the battery plan",
    "ours": "ours: forecast + plan",
}


def ladder(res: Dict) -> Path:
    table = res["ladder"]["table"]
    order = list(LABELS)
    W, H = 960, 96 + 44 * len(order) + 40
    cases = res["ladder"]["cases"]
    svg = Svg(W, H, "Score over the template, same seeds and devices",
              f"{cases} graded runs per strategy (seeds {res['seeds'][0]}-{res['seeds'][1]}, "
              f"4 devices). Bar = mean, tick = worst run.")
    L, R = 330, 120
    lo = min(min(table[s]["worst_vs_template_pct"] for s in order), 0) - 5
    hi = max(table[s]["vs_template_pct"] for s in order) + 8
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    top = 92
    for g in range(-20, int(hi) + 1, 10):
        if g < lo:
            continue
        svg.add(f'<line x1="{x(g):.1f}" x2="{x(g):.1f}" y1="{top - 8}" y2="{H - 34}" '
                f'stroke="{GRID}" stroke-width="1"/>')
        svg.text(x(g), H - 16, f"{g:+d}%" if g else "0", 11, MUTED, "middle", mono=True)
    for i, s in enumerate(order):
        v = table[s]
        y = top + i * 44
        is_ours = s.startswith("ours")
        color = US if s == "ours" else ("#2f7d8c" if is_ours else NEUTRAL)
        svg.text(L - 14, y + 19, LABELS[s], 13, INK if is_ours else INK2, "end",
                 weight=700 if s == "ours" else 400)
        x0, x1 = x(0), x(v["vs_template_pct"])
        left, width = min(x0, x1), max(2.0, abs(x1 - x0))
        svg.add(f'<rect x="{left:.1f}" y="{y + 4}" width="{width:.1f}" height="24" rx="4" fill="{color}"/>')
        wx = x(v["worst_vs_template_pct"])
        svg.add(f'<line x1="{wx:.1f}" x2="{wx:.1f}" y1="{y + 1}" y2="{y + 31}" stroke="{INK}" '
                f'stroke-width="2" opacity="0.75"/>')
        svg.text(max(x1, x0) + 10, y + 21, f"{v['vs_template_pct']:+.1f}%", 13, INK, weight=700, mono=True)
    return svg.save("ladder.svg")


# ---------------------------------------------------------------------------
# 2. Anatomy: where the rounds and the points go
# ---------------------------------------------------------------------------
def anatomy(res: Dict) -> Path:
    an = res["anatomy"]
    kinds = [("alone", "every bot asleep", "#7fd8e6"), ("thin", "1-2 bots asleep", "#2eb3c8"),
             ("crowded", "all bots awake", "#0b6f80"), ("resting", "we rest", "#2a3350")]
    W, H = 960, 330
    svg = Svg(W, H, "Where the 60 rounds go, and what each kind is worth",
              "Mean per graded run. Our rests are planned; the template's are not.")
    L, R, top = 150, 40, 100
    x = lambda v: L + v / 60 * (W - L - R)
    for i, (who, label) in enumerate((("ours", "ours"), ("template", "template"))):
        y = top + i * 90
        svg.text(L - 14, y + 24, label, 14, INK, "end", weight=700)
        svg.text(L - 14, y + 42, f"score {an[who]['score']:.1f}", 12, INK2, "end", mono=True)
        acc = 0.0
        for k, desc, color in kinds:
            n = an[who]["kinds"][k]["rounds"]
            if n <= 0:
                continue
            x0, x1 = x(acc), x(acc + n)
            svg.add(f'<rect x="{x0 + 1:.1f}" y="{y}" width="{max(1.0, x1 - x0 - 2):.1f}" height="40" '
                    f'rx="4" fill="{color}"/>')
            per = an[who]["kinds"][k]["per_round"]
            if x1 - x0 > 54:
                ink = SURFACE if k in ("alone", "thin") else INK
                svg.text((x0 + x1) / 2, y + 17, f"{n:.1f} rounds", 11, ink, "middle", weight=700, mono=True)
                svg.text((x0 + x1) / 2, y + 32, f"{per:.2f}/round" if k != "resting" else "0", 11,
                         ink, "middle", mono=True)
            else:
                # Too thin to label inside: call it out above the bar.
                svg.add(f'<line x1="{(x0 + x1) / 2:.1f}" x2="{(x0 + x1) / 2:.1f}" y1="{y - 2}" '
                        f'y2="{y - 10}" stroke="{INK2}"/>')
                svg.text(x0, y - 14, f"{n:.1f} rounds alone, {per:.2f} each", 11, INK2, mono=True)
            acc += n
    ly = H - 40
    lx = L
    for k, desc, color in kinds:
        svg.add(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="3" fill="{color}"/>')
        svg.text(lx + 18, ly, desc, 12, INK2)
        lx += 190
    return svg.save("anatomy.svg")


# ---------------------------------------------------------------------------
# 3. Nap forecast accuracy by horizon
# ---------------------------------------------------------------------------
def forecast(res: Dict) -> Path:
    acc = {int(k): v for k, v in res["anatomy"]["nap_forecast_accuracy"].items()}
    hs = sorted(acc)
    W, H = 960, 330
    svg = Svg(W, H, "How far ahead we know when a bot will nap",
              "Share of (bot, round) awake/asleep predictions that came true, by rounds ahead.")
    L, R, T, B = 70, 40, 92, 48
    lo = 50.0
    x = lambda h: L + (h - 1) / max(1, hs[-1] - 1) * (W - L - R)
    y = lambda v: T + (100 - v) / (100 - lo) * (H - T - B)
    for g in (50, 60, 70, 80, 90, 100):
        svg.add(f'<line x1="{L}" x2="{W - R}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="{GRID}"/>')
        svg.text(L - 10, y(g) + 4, f"{g}%", 11, MUTED, "end", mono=True)
    for h in hs:
        svg.text(x(h), H - 22, str(h), 11, MUTED, "middle", mono=True)
    svg.text((L + W - R) / 2, H - 6, "rounds ahead", 11, MUTED, "middle")
    pts = [(x(h), y(100 * acc[h])) for h in hs]
    svg.add(polyline(pts, US, 2.5))
    for (px, py), h in zip(pts, hs):
        svg.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{US}" stroke="{SURFACE}" stroke-width="2"/>')
    for h in (1, 5, hs[-1]):
        if h in acc:
            svg.text(x(h), y(100 * acc[h]) - 12, f"{100 * acc[h]:.0f}%", 12, INK, "middle", weight=700, mono=True)
    return svg.save("forecast.svg")


# ---------------------------------------------------------------------------
# 4-5. One run: battery traces and the score race
# ---------------------------------------------------------------------------
def one_run(seed: int = 2, team: str = "team-0") -> Tuple[Path, Path]:
    r = sim.run(seed, team, "ours", keep_rounds=True)
    teams = [team] + sorted(t for t in r.rounds[0]["nodes"] if t != team)
    T = len(r.rounds)

    def frame(title, sub, key, ymax, fmt):
        W, H = 960, 360
        svg = Svg(W, H, title, sub)
        L, R, Tm, B = 56, 150, 84, 40
        x = lambda i: L + i / T * (W - L - R)
        y = lambda v: Tm + (1 - v / ymax) * (H - Tm - B)
        steps = 4
        for g in range(steps + 1):
            v = ymax * g / steps
            svg.add(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>')
            svg.text(L - 10, y(v) + 4, fmt(v), 11, MUTED, "end", mono=True)
        for i in range(0, T + 1, 10):
            svg.text(x(i), H - 16, str(i), 11, MUTED, "middle", mono=True)
        return svg, x, y

    # Battery
    svg, x, y = frame("Battery: theirs by accident, ours by plan",
                      f"One graded run (seed {seed}). Below the dashed line a node sits the round out. "
                      "naive-max runs under even-split: same device, near-identical drain.",
                      "battery", 1.0, lambda v: f"{v:.2f}")
    svg.add(f'<line x1="{x(0):.1f}" x2="{x(T):.1f}" y1="{y(0.05):.1f}" y2="{y(0.05):.1f}" '
            f'stroke="#ff5c6c" stroke-dasharray="5 5" opacity="0.8"/>')
    ends = []
    for t in reversed(teams):
        pts = [(x(row["round"]), y(row["nodes"][t]["battery"])) for row in r.rounds]
        color = US if t == team else COLORS.get(t, NEUTRAL)
        svg.add(polyline(pts, color, 3.0 if t == team else 1.6))
        ends.append((t, pts[-1][1], color))
    _end_labels(svg, ends, x(T) + 10, team)
    p1 = svg.save("battery.svg")

    # Score race
    top = max(row["nodes"][t]["score"] for row in r.rounds for t in teams) * 1.08
    svg, x, y = frame("Score race", "Cumulative utility in the same run.", "score", top,
                      lambda v: f"{v:.0f}")
    ends = []
    for t in reversed(teams):
        pts = [(x(0), y(0))] + [(x(row["round"]), y(row["nodes"][t]["score"])) for row in r.rounds]
        color = US if t == team else COLORS.get(t, NEUTRAL)
        svg.add(polyline(pts, color, 3.0 if t == team else 1.6))
        ends.append((t, pts[-1][1], color, r.rounds[-1]["nodes"][t]["score"]))
    _end_labels(svg, [(t, yy, c) for t, yy, c, _ in ends], x(T) + 10, team,
                {t: s for t, _, _, s in ends})
    p2 = svg.save("race.svg")
    return p1, p2


def _end_labels(svg: Svg, ends, lx: float, us: str, values: Dict[str, float] = None) -> None:
    """Direct labels at the line ends, nudged apart so they never collide."""
    ends = sorted(ends, key=lambda e: e[1])
    placed: List[float] = []
    for t, yy, color in ends:
        yy = max(yy, placed[-1] + 15) if placed else yy
        placed.append(yy)
        label = short(t, us) + (f"  {values[t]:.1f}" if values else "")
        svg.add(f'<rect x="{lx}" y="{yy - 5:.1f}" width="10" height="3" rx="1.5" fill="{color}"/>')
        svg.text(lx + 15, yy + 4, label, 12, INK if t == us else INK2, weight=700 if t == us else 400)


def main() -> int:
    res = json.loads(RESULTS.read_text())
    for p in (ladder(res), anatomy(res), forecast(res), *one_run()):
        print(p.relative_to(sim.ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
