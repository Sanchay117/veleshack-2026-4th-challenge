"""
In-process arena simulator, for measuring strategies instead of guessing.

Drives the real `arena.state.Arena` object directly - the same registry,
allocation, battery and scoring code the HTTP arena runs - but without the
network, the clock or the fault injection. A 60-round graded run takes well
under a second, so an idea can be checked against hundreds of seeds and devices
before it goes anywhere near the live arena.

It reproduces the grading protocol: the three baselines on our device, and the
counterfactual anchor T (the template's strategy in our seat, same seed, same
device) so the strategy score can be computed the way the organisers do.

    python lab/sim.py --seeds 20 --strategy ours

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent-template"))
sys.path.insert(0, str(ROOT / "baselines"))

from arena.config import ScenarioConfig  # noqa: E402
from arena.state import Arena  # noqa: E402

import bot as baseline_bots  # noqa: E402
import strategy as ours  # noqa: E402
from opponents import clamp  # noqa: E402

RESOURCES = ("compute", "energy", "security")


# ---------------------------------------------------------------------------
# Seats: something that can play a node
# ---------------------------------------------------------------------------
class FnSeat:
    """A stateless template-style function: decide_bid(budget, prices, ...)."""

    def __init__(self, fn: Callable[..., Dict[str, float]]) -> None:
        self.fn = fn

    def observe(self, payload, swarm, me) -> None:
        pass

    def decide(self, payload, profile, history, total_rounds):
        return self.fn(budget=payload["budget"], prices=payload["prices"],
                       capacities=payload["capacities"], profile=profile,
                       history=history)


class BrainSeat:
    """Our strategist, given the same context the live agent gives it."""

    def __init__(self, **kwargs) -> None:
        self.brain = ours.Strategist(**kwargs)
        self.trace: List[dict] = []

    def observe(self, payload, swarm, me) -> None:
        self.brain.observe(payload, swarm, me)

    def decide(self, payload, profile, history, total_rounds):
        bid = self.brain.decide(payload, profile, history, total_rounds)
        self.trace.append(dict(self.brain.last))
        return bid


def taper_fn(budget, prices, capacities, profile, history):
    """A strong, simple heuristic: the primer's sketch done carefully.

    Our stand-in for the organisers' (unpublished) reference agent: energy
    tapered by charge from full, floors bought against a smoothed field.
    """
    w = profile["weights"]
    battery = float((profile.get("features") or {}).get("battery", 1.0))
    keep = max(0.0, min(1.0, (battery - 0.15) / 0.85)) ** 1.5
    bid = {k: budget * w[k] for k in RESOURCES}
    freed = bid["energy"] * (1 - keep)
    bid["energy"] -= freed
    cs = w["compute"] + w["security"]
    bid["compute"] += freed * w["compute"] / cs
    bid["security"] += freed * w["security"] / cs
    others = baseline_bots.estimate_others(history)
    return baseline_bots.buy_floors(
        bid, budget, prices, capacities,
        baseline_bots._floor_target(profile["q_min"], capacities["compute"]),
        baseline_bots._floor_target(profile["s_min"], capacities["security"]),
        others)


SEATS: Dict[str, Callable[[], Any]] = {
    "template": lambda: FnSeat(baseline_bots.naive_max),
    "taper": lambda: FnSeat(taper_fn),
    "ours": lambda: BrainSeat(),
    "ours-no-model": lambda: BrainSeat(use_opponent_model=False),
    "ours-no-plan": lambda: BrainSeat(use_planner=False),
}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    seed: int
    team: str
    strategy: str
    score: float
    bots: Dict[str, float]
    idle: int
    floors: int
    lsw: List[float]
    rounds: List[dict] = field(default_factory=list)
    think_ms: float = 0.0


def run(seed: int, team: str, seat_name: str, scenario: str = "graded",
        keep_rounds: bool = False) -> RunResult:
    cfg = ScenarioConfig.load(scenario)
    cfg.seed = seed
    arena = Arena(cfg)
    bot_nodes = {}
    for b in cfg.baselines:
        node = arena.register(f"bot-{b}")
        bot_nodes[node.node_id] = (node, baseline_bots.STRATEGIES[b], [])
    me = arena.register(team)
    seat = SEATS[seat_name]()
    my_history: List[dict] = []
    rounds_log: List[dict] = []
    t_think = 0.0

    for _ in range(cfg.total_rounds):
        state = arena.open_round()
        pub = state.public()

        for node, fn, hist in bot_nodes.values():
            if node.admissible(cfg.battery_cutoff, cfg.kappa_bar):
                bid = fn(budget=node.budget, prices=pub["prices"],
                         capacities=pub["capacities"], profile=node.profile(),
                         history=hist)
                arena.submit(node, state.index, clamp(bid, node.budget))

        admissible = me.admissible(cfg.battery_cutoff, cfg.kappa_bar)
        payload = {**pub, "budget": me.budget,
                   "you": {"node_id": me.node_id, "admissible": admissible,
                           "battery": round(me.battery, 4),
                           "compromise": me.compromise, "already_submitted": False}}
        seat.observe(payload, arena.swarm(), me.node_id)
        if admissible:
            t = time.perf_counter()
            profile = me.profile()
            bid = seat.decide(payload, profile, my_history, cfg.total_rounds)
            t_think += time.perf_counter() - t
            arena.submit(me, state.index, clamp(bid, me.budget))

        arena.settle()
        for node, fn, hist in bot_nodes.values():
            res = state.results.get(node.node_id)
            if res:
                hist.append(res)
        res = state.results.get(me.node_id)
        if res:
            my_history.append(res)
        if keep_rounds:
            rounds_log.append({
                "round": state.index,
                "lsw": state.lsw,
                "nodes": {n.team: {"battery": n.battery, "score": n.score,
                                   "awake": n.node_id in state.results,
                                   "utility": state.results.get(n.node_id, {}).get("utility", 0.0)}
                          for n in arena.nodes.values()},
                "brain": (seat.trace[-1] if isinstance(seat, BrainSeat) and seat.trace
                          and seat.trace[-1].get("round") == state.index else None),
            })

    return RunResult(
        seed=seed, team=team, strategy=seat_name, score=me.score,
        bots={n.team: n.score for n, _, _ in bot_nodes.values()},
        idle=me.rounds_idle, floors=me.floor_violations,
        lsw=[r.lsw for r in arena.rounds if r.settled],
        rounds=rounds_log,
        think_ms=1000 * t_think / max(1, me.rounds_participated),
    )


# ---------------------------------------------------------------------------
# Many runs
# ---------------------------------------------------------------------------
def compare(seeds: List[int], teams: List[str], strategies: List[str],
            scenario: str = "graded") -> Dict[str, Any]:
    rows = []
    for seed in seeds:
        for team in teams:
            base = run(seed, team, "template", scenario)
            for s in strategies:
                r = base if s == "template" else run(seed, team, s, scenario)
                rows.append({
                    "seed": seed, "team": team, "strategy": s, "score": r.score,
                    "T": base.score, "best_bot": max(r.bots.values()),
                    "idle": r.idle, "floors": r.floors,
                    "lsw": statistics.fmean(r.lsw) if r.lsw else 0.0,
                    "think_ms": r.think_ms,
                })
    summary = {}
    for s in strategies:
        sub = [x for x in rows if x["strategy"] == s]
        summary[s] = {
            "mean_score": statistics.fmean(x["score"] for x in sub),
            "vs_template_pct": 100 * statistics.fmean(x["score"] / x["T"] - 1 for x in sub),
            "vs_best_bot_pct": 100 * statistics.fmean(x["score"] / x["best_bot"] - 1 for x in sub),
            "beats_all_bots": sum(x["score"] > x["best_bot"] for x in sub) / len(sub),
            "idle": statistics.fmean(x["idle"] for x in sub),
            "floors": statistics.fmean(x["floors"] for x in sub),
            "lsw": statistics.fmean(x["lsw"] for x in sub),
            "think_ms": statistics.fmean(x["think_ms"] for x in sub),
        }
    return {"rows": rows, "summary": summary}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare strategies in-process.")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--teams", type=int, default=3)
    p.add_argument("--strategy", nargs="+", default=["template", "taper", "ours"])
    p.add_argument("--scenario", default="graded")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    seeds = list(range(a.first_seed, a.first_seed + a.seeds))
    teams = [f"team-{i}" for i in range(a.teams)]
    t = time.time()
    out = compare(seeds, teams, a.strategy, a.scenario)
    print(f"{len(seeds)} seeds x {len(teams)} devices, {time.time() - t:.1f}s\n")
    print(f"{'strategy':<16}{'score':>8}{'vs T':>9}{'vs best bot':>13}{'beats all':>11}"
          f"{'idle':>7}{'floors':>8}{'LSW':>8}{'ms':>7}")
    for s, v in out["summary"].items():
        print(f"{s:<16}{v['mean_score']:>8.2f}{v['vs_template_pct']:>8.1f}%"
              f"{v['vs_best_bot_pct']:>12.1f}%{v['beats_all_bots']:>10.0%}"
              f"{v['idle']:>7.1f}{v['floors']:>8.1f}{v['lsw']:>8.2f}{v['think_ms']:>7.1f}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
