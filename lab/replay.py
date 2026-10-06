"""
Replay a simulated run through the live dashboard.

Runs one game in-process (see sim.py), records the exact telemetry the live
agent would have served after every round, then serves the dashboard and plays
those frames back at any speed. Same page, same data shape as a live run, so
it is both the quickest way to look at a strategy and a demo that cannot be
spoiled by a slow network.

    python lab/replay.py --seed 2 --team team-0 --port 8090 --speed 0.8
    open http://localhost:8090

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import sim
from telemetry import DASHBOARD, Telemetry

from arena.config import ScenarioConfig
from arena.state import Arena
from opponents import clamp
from runner import clamp as runner_clamp
import bot as baseline_bots


def record(seed: int, team: str, scenario: str = "graded") -> List[Dict[str, Any]]:
    """Play one run and return a telemetry snapshot per round."""
    cfg = ScenarioConfig.load(scenario)
    cfg.seed = seed
    arena = Arena(cfg)
    bots = []
    for b in cfg.baselines:
        node = arena.register(f"bot-{b}")
        bots.append((node, baseline_bots.STRATEGIES[b], []))
    me = arena.register(team)
    seat = sim.BrainSeat()
    tel = Telemetry(team)
    history: List[dict] = []
    frames: List[Dict[str, Any]] = []

    def board() -> Dict[str, Any]:
        return {"leaderboard": arena.leaderboard(), "status": {**arena.status(), "scenario": cfg.name}}

    for _ in range(cfg.total_rounds):
        state = arena.open_round()
        pub = state.public()
        tel.on_board(state.index, board())
        for node, fn, hist in bots:
            if node.admissible(cfg.battery_cutoff, cfg.kappa_bar):
                bid = fn(budget=node.budget, prices=pub["prices"], capacities=pub["capacities"],
                         profile=node.profile(), history=hist)
                arena.submit(node, state.index, runner_clamp(bid, node.budget))
        ok = me.admissible(cfg.battery_cutoff, cfg.kappa_bar)
        payload = {**pub, "budget": me.budget,
                   "you": {"admissible": ok, "battery": round(me.battery, 4)}}
        seat.observe(payload, arena.swarm(), me.node_id)
        if ok:
            bid = seat.decide(payload, me.profile(), history, cfg.total_rounds)
            tel.on_decision(seat.brain.last)
            arena.submit(me, state.index, clamp(bid, me.budget, down=True))
        else:
            seat.brain.note_rest(payload)
            tel.on_decision(seat.brain.last)
        frames.append(copy.deepcopy(tel.snapshot()))
        arena.settle()
        for node, fn, hist in bots:
            if node.node_id in state.results:
                hist.append(state.results[node.node_id])
        if me.node_id in state.results:
            history.append(state.results[me.node_id])

    tel.on_board(cfg.total_rounds + 1, board())
    final = copy.deepcopy(tel.snapshot())
    final["decision"] = {"round": cfg.total_rounds + 1, "final": True}
    frames.append(final)
    return frames


def serve(frames: List[Dict[str, Any]], port: int, speed: float, hold: float) -> None:
    start = time.monotonic()
    period = speed * len(frames) + hold

    def current() -> Dict[str, Any]:
        t = (time.monotonic() - start) % period
        return frames[min(len(frames) - 1, int(t / speed))]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/api/state"):
                body, ctype = json.dumps(current(), default=float).encode(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, ctype = DASHBOARD.read_bytes(), "text/html; charset=utf-8"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a: Any) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"replaying {len(frames)} rounds on http://localhost:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="Replay a simulated run in the dashboard.")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--team", default="team-0")
    p.add_argument("--scenario", default="graded")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--speed", type=float, default=0.8, help="seconds per round")
    p.add_argument("--hold", type=float, default=12.0, help="seconds to hold the final frame")
    p.add_argument("--dump", default=None, help="also write the frames to this JSON file")
    a = p.parse_args()
    frames = record(a.seed, a.team, a.scenario)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as fh:
            json.dump(frames, fh, default=float)
    serve(frames, a.port, a.speed, a.hold)
    return 0


if __name__ == "__main__":
    threading.current_thread().name = "replay"
    raise SystemExit(main())
