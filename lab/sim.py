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
import multiprocessing
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent-template"))
sys.path.insert(0, str(ROOT / "baselines"))

from arena.config import ScenarioConfig  # noqa: E402
from arena.state import Arena  # noqa: E402

import bot as baseline_bots  # noqa: E402
import strategy as ours  # noqa: E402
from opponents import clamp  # noqa: E402
from runner import clamp as runner_clamp  # noqa: E402

RESOURCES = ("compute", "energy", "security")

# Experiments measure strategy, so the live agent's think-time guard is lifted:
# results must not depend on how busy the machine running them is.
ours.THINK_BUDGET = 1e9


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


class MyopicSeat(BrainSeat):
    """Exact single-round best response to the known field, blind to the battery.

    Isolates what the primer calls the Kelly best response: the optimal split
    for this round alone, with charge treated as free.
    """

    def __init__(self) -> None:
        super().__init__(use_planner=False)

    def decide(self, payload, profile, history, total_rounds):
        saved = ours.FIXED_SHADOW_PRICE
        ours.FIXED_SHADOW_PRICE = 0.0
        try:
            return super().decide(payload, profile, history, total_rounds)
        finally:
            ours.FIXED_SHADOW_PRICE = saved


SEATS: Dict[str, Callable[[], Any]] = {
    "template": lambda: FnSeat(baseline_bots.naive_max),
    "even-split": lambda: FnSeat(baseline_bots.even_split),
    "proportional": lambda: FnSeat(baseline_bots.proportional),
    "best-response": lambda: MyopicSeat(),
    "taper": lambda: FnSeat(taper_fn),
    "ours": lambda: BrainSeat(),
    "ours-no-model": lambda: BrainSeat(use_opponent_model=False),
    "ours-no-plan": lambda: BrainSeat(use_planner=False),
    "ours-look": lambda: BrainSeat(lookahead=4),
    "ours-sampled": lambda: BrainSeat(scenarios=6),
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


@dataclass
class Player:
    team: str
    seat: Any
    node: Any = None
    history: List[dict] = field(default_factory=list)
    think: float = 0.0


def _play(arena: Arena, cfg: ScenarioConfig, p: Player, state, pub: dict) -> None:
    """One player's turn: observe the round, and bid if allowed to."""
    node = p.node
    ok = node.admissible(cfg.battery_cutoff, cfg.kappa_bar)
    payload = {**pub, "budget": node.budget,
               "you": {"node_id": node.node_id, "admissible": ok,
                       "battery": round(node.battery, 4),
                       "compromise": node.compromise, "already_submitted": False}}
    p.seat.observe(payload, arena.swarm(), node.node_id)
    if ok:
        t = time.perf_counter()
        bid = p.seat.decide(payload, node.profile(), p.history, cfg.total_rounds)
        p.think += time.perf_counter() - t
        # Each player's bid goes through the same last step as in the live
        # system: the bots' runner clamps without rounding, ours rounds down.
        legal = runner_clamp(bid, node.budget) if isinstance(p.seat, FnSeat) \
            else clamp(bid, node.budget, down=True)
        arena.submit(node, state.index, legal)


def run(seed: int, team: str, seat_name: str, scenario: str = "graded",
        keep_rounds: bool = False, overrides: Optional[Dict[str, Any]] = None,
        strangers: Sequence[str] = (), join_round: int = 1) -> RunResult:
    """Play one run with `seat_name` in our seat.

    overrides   scenario fields to change (e.g. battery_drain) - physics the
                agent has never been told about
    strangers   extra non-baseline players (seat names) the opponent model
                has no shadow for
    join_round  the round our agent registers in; the bots play alone before
    """
    cfg = ScenarioConfig.load(scenario)
    cfg.seed = seed
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    # No wall clock here: a round stays open until everyone has played, so a
    # loaded machine measures strategy, not CPU contention. (Think time is
    # reported separately and guarded in the agent itself.)
    cfg.round_seconds = 3600.0
    arena = Arena(cfg)
    bots = [Player(f"bot-{b}", FnSeat(baseline_bots.STRATEGIES[b])) for b in cfg.baselines]
    for p in bots:
        p.node = arena.register(p.team)
    me = Player(team, SEATS[seat_name]())
    others = [Player(f"stranger-{i}-{s}", SEATS[s]()) for i, s in enumerate(strangers)]
    rounds_log: List[dict] = []

    for index in range(1, cfg.total_rounds + 1):
        if index == join_round:
            me.node = arena.register(team)
            for p in others:
                p.node = arena.register(p.team)
        state = arena.open_round()
        pub = state.public()
        for p in bots + others + [me]:
            if p.node is not None:
                _play(arena, cfg, p, state, pub)
        arena.settle()
        for p in bots + others + [me]:
            if p.node is not None and p.node.node_id in state.results:
                p.history.append(state.results[p.node.node_id])
        if keep_rounds:
            seat = me.seat
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
        seed=seed, team=team, strategy=seat_name, score=me.node.score,
        bots={p.team: p.node.score for p in bots + others},
        idle=me.node.rounds_idle, floors=me.node.floor_violations,
        lsw=[r.lsw for r in arena.rounds if r.settled],
        rounds=rounds_log,
        think_ms=1000 * me.think / max(1, me.node.rounds_participated),
    )


# ---------------------------------------------------------------------------
# Many runs
# ---------------------------------------------------------------------------
def _one_case(args) -> List[dict]:
    """Every strategy on one (seed, device), plus the template anchor T."""
    seed, team, strategies, scenario, kw = args
    base = run(seed, team, "template", scenario, **kw)
    rows = []
    for s in strategies:
        r = base if s == "template" else run(seed, team, s, scenario, **kw)
        rows.append({
            "seed": seed, "team": team, "strategy": s, "score": r.score,
            "T": base.score, "best_bot": max(v for k, v in r.bots.items() if k.startswith("bot-")),
            "bots": r.bots, "bots_under_T": base.bots,
            "idle": r.idle, "floors": r.floors,
            "lsw": statistics.fmean(r.lsw) if r.lsw else 0.0,
            "think_ms": r.think_ms,
        })
    return rows


def compare(seeds: List[int], teams: List[str], strategies: List[str],
            scenario: str = "graded", jobs: int = 1, **kw) -> Dict[str, Any]:
    cases = [(seed, team, strategies, scenario, kw) for seed in seeds for team in teams]
    if jobs > 1:
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            chunks = pool.map(_one_case, cases)
    else:
        chunks = [_one_case(c) for c in cases]
    rows = [r for chunk in chunks for r in chunk]
    summary = {}
    for s in strategies:
        sub = [x for x in rows if x["strategy"] == s]
        summary[s] = {
            "mean_score": statistics.fmean(x["score"] for x in sub),
            "vs_template_pct": 100 * statistics.fmean(x["score"] / x["T"] - 1 for x in sub),
            "worst_vs_template_pct": 100 * min(x["score"] / x["T"] - 1 for x in sub),
            "vs_best_bot_pct": 100 * statistics.fmean(x["score"] / x["best_bot"] - 1 for x in sub),
            "beats_all_bots": sum(x["score"] > x["best_bot"] for x in sub) / len(sub),
            "idle": statistics.fmean(x["idle"] for x in sub),
            "floors": statistics.fmean(x["floors"] for x in sub),
            "lsw": statistics.fmean(x["lsw"] for x in sub),
            "think_ms": statistics.fmean(x["think_ms"] for x in sub),
        }
    return {"rows": rows, "summary": summary}


def print_summary(out: Dict[str, Any]) -> None:
    print(f"{'strategy':<16}{'score':>8}{'vs T':>9}{'worst':>8}{'vs best bot':>13}"
          f"{'beats all':>11}{'idle':>7}{'floors':>8}{'LSW':>8}{'ms':>7}")
    for s, v in out["summary"].items():
        print(f"{s:<16}{v['mean_score']:>8.2f}{v['vs_template_pct']:>8.1f}%"
              f"{v['worst_vs_template_pct']:>7.1f}%"
              f"{v['vs_best_bot_pct']:>12.1f}%{v['beats_all_bots']:>10.0%}"
              f"{v['idle']:>7.1f}{v['floors']:>8.1f}{v['lsw']:>8.2f}{v['think_ms']:>7.1f}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare strategies in-process.")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--teams", type=int, default=3, help="number of generated devices")
    p.add_argument("--team", nargs="+", default=None, help="explicit team names instead")
    p.add_argument("--strategy", nargs="+", default=["template", "taper", "ours"])
    p.add_argument("--scenario", default="graded")
    p.add_argument("--drain", type=float, default=None, help="override battery_drain")
    p.add_argument("--recharge", type=float, default=None, help="override recharge_rate")
    p.add_argument("--rounds", type=int, default=None, help="override total_rounds")
    p.add_argument("--stranger", nargs="*", default=[], help="extra unmodelled players")
    p.add_argument("--join", type=int, default=1, help="round our agent joins in")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    overrides = {k: v for k, v in (("battery_drain", a.drain), ("recharge_rate", a.recharge),
                                   ("total_rounds", a.rounds)) if v is not None}
    seeds = list(range(a.first_seed, a.first_seed + a.seeds))
    teams = a.team or [f"team-{i}" for i in range(a.teams)]
    t = time.time()
    out = compare(seeds, teams, a.strategy, a.scenario, jobs=a.jobs,
                  overrides=overrides, strangers=a.stranger, join_round=a.join)
    print(f"{len(seeds)} seeds x {len(teams)} devices, {time.time() - t:.1f}s"
          + (f"  overrides={overrides}" if overrides else "")
          + (f"  strangers={a.stranger}" if a.stranger else "")
          + (f"  join_round={a.join}" if a.join > 1 else "") + "\n")
    print_summary(out)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
