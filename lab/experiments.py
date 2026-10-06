"""
Every number in the write-up, reproducible.

    python lab/experiments.py                 # ~5 minutes on a laptop
    python lab/experiments.py --quick         # a smaller sample, for checking

Writes docs/results/results.json and prints the tables the README quotes.
All runs use the graded scenario with seeds the agent has never been tuned
on (1000+), on four devices including our own team's.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import sim

OUT = sim.ROOT / "docs" / "results" / "results.json"
TEAMS = ["NullPointerException", "team-a", "team-b", "team-c"]
LADDER = ["template", "even-split", "proportional", "best-response", "taper",
          "ours-no-model", "ours-no-plan", "ours"]


def pct(x: float) -> float:
    return round(100.0 * x, 2)


# ---------------------------------------------------------------------------
# 1. The ladder: every strategy in our seat, same seeds, same devices
# ---------------------------------------------------------------------------
def ladder(seeds, jobs) -> Dict[str, Any]:
    out = sim.compare(seeds, TEAMS, LADDER, jobs=jobs)
    rows = out["rows"]
    table = {}
    for s in LADDER:
        sub = [r for r in rows if r["strategy"] == s]
        rel = [r["score"] / r["T"] - 1 for r in sub]
        table[s] = {
            "mean_score": round(statistics.fmean(r["score"] for r in sub), 3),
            "vs_template_pct": pct(statistics.fmean(rel)),
            "worst_vs_template_pct": pct(min(rel)),
            "p10_vs_template_pct": pct(sorted(rel)[len(rel) // 10]),
            "beats_all_bots": round(sum(r["score"] > r["best_bot"] for r in sub) / len(sub), 4),
            "rested": round(statistics.fmean(r["idle"] for r in sub), 2),
            "floor_misses": round(statistics.fmean(r["floors"] for r in sub), 2),
            "lsw": round(statistics.fmean(r["lsw"] for r in sub), 3),
            "think_ms": round(statistics.fmean(r["think_ms"] for r in sub), 1),
        }
    # Did we take from the swarm? Bots' scores with us in the seat vs with
    # the template in the seat, same seed and device.
    ours = [r for r in rows if r["strategy"] == "ours"]
    bots_with_us = statistics.fmean(sum(r["bots"].values()) for r in ours)
    bots_with_T = statistics.fmean(sum(r["bots_under_T"].values()) for r in ours)
    swarm_with_us = statistics.fmean(sum(r["bots"].values()) + r["score"] for r in ours)
    swarm_with_T = statistics.fmean(sum(r["bots_under_T"].values()) + r["T"] for r in ours)
    welfare = {
        "bots_total_with_template": round(bots_with_T, 3),
        "bots_total_with_ours": round(bots_with_us, 3),
        "bots_change_pct": pct(bots_with_us / bots_with_T - 1),
        "swarm_total_with_template": round(swarm_with_T, 3),
        "swarm_total_with_ours": round(swarm_with_us, 3),
        "swarm_change_pct": pct(swarm_with_us / swarm_with_T - 1),
        "lsw_with_template": table["template"]["lsw"],
        "lsw_with_ours": table["ours"]["lsw"],
    }
    return {"cases": len(rows) // len(LADDER), "table": table, "welfare": welfare,
            "rows": [{k: v for k, v in r.items() if k not in ("bots", "bots_under_T")}
                     for r in rows]}


# ---------------------------------------------------------------------------
# 2. Anatomy: where the points come from, round by round
# ---------------------------------------------------------------------------
def _anatomy_case(args):
    seed, team = args
    out = {}
    for strat in ("ours", "template"):
        r = sim.run(seed, team, strat, keep_rounds=True)
        kinds = {"alone": [0, 0.0], "thin": [0, 0.0], "crowded": [0, 0.0], "resting": [0, 0.0]}
        energy = []
        horizon_hits: Dict[int, List[int]] = {}
        log = r.rounds
        for i, row in enumerate(log):
            n = row["nodes"]
            me = n[team]
            bots_awake = sum(1 for t, v in n.items() if t.startswith("bot-") and v["awake"])
            if not me["awake"]:
                k = "resting"
            elif bots_awake == 0:
                k = "alone"
            elif bots_awake < 3:
                k = "thin"
            else:
                k = "crowded"
            kinds[k][0] += 1
            kinds[k][1] += me["utility"]
            b = row["brain"]
            if strat == "ours" and b:
                energy.append(b["x_energy"])
                for h, awake in enumerate(b.get("forecast_awake") or [], start=1):
                    if i + h >= len(log) or h > 12:
                        break
                    actual = log[i + h]["nodes"]
                    for t, v in actual.items():
                        if t.startswith("bot-"):
                            horizon_hits.setdefault(h, []).append(int((t in awake) == v["awake"]))
        out[strat] = {"kinds": kinds, "score": r.score,
                      "mean_x_energy": statistics.fmean(energy) if energy else 0.0,
                      "forecast": {h: statistics.fmean(v) for h, v in horizon_hits.items()}}
    return out


def anatomy(seeds, jobs) -> Dict[str, Any]:
    cases = [(s, t) for s in seeds for t in TEAMS[:2]]
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        res = pool.map(_anatomy_case, cases)
    summary = {}
    for strat in ("ours", "template"):
        kinds = {}
        for k in ("alone", "thin", "crowded", "resting"):
            rounds = statistics.fmean(r[strat]["kinds"][k][0] for r in res)
            util = statistics.fmean(r[strat]["kinds"][k][1] for r in res)
            kinds[k] = {"rounds": round(rounds, 2), "utility": round(util, 3),
                        "per_round": round(util / rounds, 3) if rounds else 0.0}
        summary[strat] = {"kinds": kinds,
                          "score": round(statistics.fmean(r[strat]["score"] for r in res), 3)}
    summary["ours"]["mean_x_energy"] = round(statistics.fmean(r["ours"]["mean_x_energy"] for r in res), 4)
    horizons = sorted({h for r in res for h in r["ours"]["forecast"]})
    summary["nap_forecast_accuracy"] = {
        h: round(statistics.fmean(r["ours"]["forecast"][h] for r in res if h in r["ours"]["forecast"]), 4)
        for h in horizons}
    return {"cases": len(cases), **summary}


# ---------------------------------------------------------------------------
# 3. Robustness and refinements
# ---------------------------------------------------------------------------
def stress(seeds, jobs) -> Dict[str, Any]:
    cases = {
        "graded (baseline)": {},
        "battery drains 50% faster, recharges 32% slower": {"overrides": {"battery_drain": 0.45, "recharge_rate": 0.15}},
        "battery drains 33% slower": {"overrides": {"battery_drain": 0.20}},
        "unknown extra agent in the swarm": {"strangers": ["taper"]},
        "joins at round 12": {"join_round": 12},
    }
    out = {}
    for name, kw in cases.items():
        res = sim.compare(seeds, TEAMS[:2], ["template", "taper", "ours"], jobs=jobs, **kw)
        out[name] = {s: {"vs_template_pct": round(v["vs_template_pct"], 2),
                         "worst_vs_template_pct": round(v["worst_vs_template_pct"], 2)}
                     for s, v in res["summary"].items()}
    res = sim.compare(seeds, TEAMS[:2], ["template", "taper", "ours"], scenario="practice", jobs=jobs)
    out["practice scenario (40 rounds)"] = {s: {"vs_template_pct": round(v["vs_template_pct"], 2),
                                                "worst_vs_template_pct": round(v["worst_vs_template_pct"], 2)}
                                            for s, v in res["summary"].items()}
    return out


def refinements(seeds, jobs) -> Dict[str, Any]:
    res = sim.compare(seeds, TEAMS[1:4], ["template", "ours", "ours-look", "ours-sampled"], jobs=jobs)
    return {s: {"vs_template_pct": round(v["vs_template_pct"], 2), "think_ms": round(v["think_ms"], 1)}
            for s, v in res["summary"].items()}


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = p.parse_args()
    n = 6 if a.quick else 40
    seeds = list(range(1000, 1000 + n))
    t0 = time.time()

    results: Dict[str, Any] = {"seeds": [seeds[0], seeds[-1]], "devices": TEAMS}
    print("ladder ...", flush=True)
    results["ladder"] = ladder(seeds, a.jobs)
    print("anatomy ...", flush=True)
    results["anatomy"] = anatomy(seeds[: max(4, n // 2)], a.jobs)
    print("stress ...", flush=True)
    results["stress"] = stress(seeds[: max(4, n // 2)], a.jobs)
    print("refinements ...", flush=True)
    results["refinements"] = refinements(seeds[: max(3, n // 4)], a.jobs)
    results["seconds"] = round(time.time() - t0, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))

    print(f"\n{results['ladder']['cases']} cases per strategy, {results['seconds']}s\n")
    print(f"{'strategy':<15}{'vs T':>8}{'worst':>8}{'p10':>8}{'beats all':>10}{'rested':>8}{'floors':>8}{'LSW':>8}")
    for s, v in results["ladder"]["table"].items():
        print(f"{s:<15}{v['vs_template_pct']:>7.1f}%{v['worst_vs_template_pct']:>7.1f}%"
              f"{v['p10_vs_template_pct']:>7.1f}%{v['beats_all_bots']:>10.0%}{v['rested']:>8.1f}"
              f"{v['floor_misses']:>8.2f}{v['lsw']:>8.2f}")
    print("\nwelfare", json.dumps(results["ladder"]["welfare"], indent=1))
    print("\nanatomy", json.dumps({k: v for k, v in results["anatomy"].items()}, indent=1))
    print("\nstress", json.dumps(results["stress"], indent=1))
    print("\nrefinements", json.dumps(results["refinements"], indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
