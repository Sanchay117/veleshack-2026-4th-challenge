"""
Unit tests for the agent: the maths, the opponent model, the planner, and the
guarantees the strategy makes about every bid it returns.

    python -m unittest discover -s tests -p 'test_*.py'      (or: make test)

Standard library only, so they run anywhere the arena runs.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import math
import random
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "agent-template", ROOT / "baselines", ROOT / "lab"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from arena import game  # noqa: E402
import bot as baselines  # noqa: E402
import econ  # noqa: E402
import opponents  # noqa: E402
import planner  # noqa: E402
import strategy  # noqa: E402
from agent import _sanitise  # noqa: E402
from client import ArenaClient  # noqa: E402

RES = ("compute", "energy", "security")


def random_profile(rng: random.Random) -> dict:
    w = [rng.uniform(0.2, 0.5), rng.uniform(0.15, 0.4)]
    w.append(max(0.1, 1 - sum(w)))
    total = sum(w)
    return {
        "weights": {k: v / total for k, v in zip(RES, w)},
        "q_min": rng.uniform(0.12, 0.22),
        "s_min": rng.uniform(0.08, 0.16),
        "features": {"battery": rng.uniform(0.06, 1.0), "mobility": rng.uniform(0, 0.45)},
    }


class EconMatchesArena(unittest.TestCase):
    """econ.py must be the arena's game.py, restated - never an approximation of it."""

    def test_share_inverts(self):
        rng = random.Random(1)
        for _ in range(500):
            S, C = rng.uniform(0.05, 3), rng.uniform(0.7, 1.3)
            t = rng.uniform(0.01, 0.9) * C
            b = econ.bid_for_share(t, S, C)
            self.assertAlmostEqual(float(econ.share(b, S, C)), t, places=9)

    def test_alone_takes_whole_pool(self):
        self.assertAlmostEqual(float(econ.share(1e-6, 0.0, 1.17)), 1.17)
        self.assertEqual(float(econ.share(0.0, 0.0, 1.17)), 0.0)

    def test_utility_and_floors_match_game(self):
        rng = random.Random(2)
        for _ in range(500):
            prof = random_profile(rng)
            x = {k: rng.uniform(0, 0.6) for k in RES}
            want = game.ces_utility(x, prof["weights"])
            want, _ = game.apply_floors(want, x, prof["q_min"], prof["s_min"])
            got = float(econ.ces(x["compute"], x["energy"], x["security"], prof["weights"]))
            got *= float(econ.floor_factor(x["compute"], x["security"], prof["q_min"], prof["s_min"]))
            self.assertAlmostEqual(got, want, places=9)


class ShadowsMatchBaselines(unittest.TestCase):
    """The opponent model reproduces baselines/bot.py bid for bid."""

    def test_every_bot_every_input(self):
        rng = random.Random(3)
        for _ in range(400):
            prof = random_profile(rng)
            budget = rng.uniform(0.75, 1.25)
            prices = {k: rng.uniform(0.01, 3) for k in RES}
            caps = {k: rng.uniform(0.7, 1.3) for k in RES}
            history = [] if rng.random() < 0.2 else [{
                "prices": {k: rng.uniform(0.01, 3) for k in RES},
                "capacities": {k: rng.uniform(0.7, 1.3) for k in RES},
                "bid": {k: rng.uniform(0, 0.5) for k in RES},
            }]
            for name, real in baselines.STRATEGIES.items():
                want = opponents.clamp(real(budget, prices, caps, prof, history), budget)
                shadow = opponents.SHADOWS[f"bot-{name}"]
                got = opponents.clamp(shadow(budget, prices, caps, prof,
                                             history[-1] if history else None), budget)
                for k in RES:
                    self.assertAlmostEqual(got[k], want[k], places=6, msg=f"{name}/{k}")


class OpponentModelInTheArena(unittest.TestCase):
    """In a full simulated run the forecast field matches the truth."""

    def test_forecast_is_exact(self):
        """Exact apart from a round or two, whichever way the bots keep history."""
        import sim
        for collect in (sim.BOT_COLLECT, 1.0):
            r = sim.run(7, "team-test", "ours", keep_rounds=True, bot_collect=collect)
            errors = [row["brain"]["model_error"] for row in r.rounds if row["brain"]]
            self.assertTrue(errors)
            self.assertLess(sum(errors) / len(errors), 0.005, msg=f"collect={collect}")
            self.assertLess(errors[-1], 1e-5)   # six-decimal rounding only
            self.assertEqual(r.floors, 0)

    def test_unknown_player_is_absorbed(self):
        import sim
        r = sim.run(7, "team-test", "ours", strangers=["taper"])
        self.assertGreater(r.score, max(v for k, v in r.bots.items() if k.startswith("bot-")))


class PlannerGuarantees(unittest.TestCase):
    def setUp(self):
        self.dev = planner.Device(weights={"compute": 0.4, "energy": 0.28, "security": 0.32},
                                  q_min=0.17, s_min=0.11, mobility=0.2)

    def menu(self, field, budget=1.0, caps=None):
        caps = caps or {k: 1.0 for k in RES}
        return planner.round_menu(planner.Market1(budget, caps, field), self.dev, 0.3, 0.004)

    def test_bids_are_legal(self):
        rng = random.Random(4)
        for _ in range(200):
            field = {k: rng.choice([0.0, rng.uniform(0.01, 3)]) for k in RES}
            budget = rng.uniform(0.75, 1.25)
            m = self.menu(field, budget)
            self.assertTrue(np.all(m.bids >= 0))
            self.assertTrue(np.all(m.bids.sum(axis=1) <= budget + 1e-9))
            self.assertTrue(np.all(np.isfinite(m.utility)))

    def test_floors_bought_when_affordable(self):
        m = self.menu({k: 1.0 for k in RES})
        res = econ.realised_utility(dict(zip(RES, m.bids[0])), {k: 1.0 for k in RES},
                                    {k: 1.0 for k in RES}, self.dev.weights,
                                    self.dev.q_min, self.dev.s_min)
        self.assertGreaterEqual(res["x_compute"], self.dev.q_min)
        self.assertGreaterEqual(res["x_security"], self.dev.s_min)

    def test_last_round_spends_the_battery(self):
        m = self.menu({k: 1.0 for k in RES})
        p = planner.plan(0.9, [m], cutoff=0.05, recharge=0.22)
        self.assertEqual(p.action, int(np.argmax(m.utility)))

    def test_never_plans_past_a_flat_battery(self):
        menus = [self.menu({k: 1.0 for k in RES}) for _ in range(30)]
        p = planner.plan(0.5, menus, cutoff=0.05, recharge=0.22)
        b = p.battery_path
        for t, a in enumerate(p.action_path):
            if b[t] <= 0.05:
                self.assertEqual(a, -1)


class StrategyGuarantees(unittest.TestCase):
    """Whatever the input, the strategy returns a legal bid, fast."""

    def test_random_payloads_without_swarm(self):
        rng = random.Random(5)
        for i in range(60):
            prof = random_profile(rng)
            payload = {"round": i + 1, "budget": rng.uniform(0.75, 1.25),
                       "prices": {k: rng.uniform(0.01, 3) for k in RES},
                       "capacities": {k: rng.uniform(0.7, 1.3) for k in RES},
                       "you": {"battery": prof["features"]["battery"]}}
            brain = strategy.Strategist()
            brain.observe(payload)
            t = time.perf_counter()
            bid = brain.decide(payload, prof, [], total_rounds=60)
            self.assertLess(time.perf_counter() - t, 1.0)
            self.assertTrue(all(v >= 0 and math.isfinite(v) for v in bid.values()))
            self.assertLessEqual(sum(bid.values()), payload["budget"] + 1e-9)

    def test_template_signature_still_works(self):
        prof = random_profile(random.Random(6))
        bid = strategy.decide_bid(1.0, {k: 1.0 for k in RES}, {k: 1.0 for k in RES}, prof, [])
        self.assertLessEqual(sum(bid.values()), 1.0 + 1e-9)

    def test_garbage_profile_falls_back_safely(self):
        bid = strategy.decide_bid(1.0, {}, {}, {"weights": {}}, [])
        self.assertTrue(all(v >= 0 for v in bid.values()))
        self.assertLessEqual(sum(bid.values()), 1.0 + 1e-9)


class Plumbing(unittest.TestCase):
    def test_sanitise(self):
        clean = _sanitise({"compute": float("nan"), "energy": -1, "security": 5}, 1.0)
        self.assertEqual(clean["compute"], 0.0)
        self.assertEqual(clean["energy"], 0.0)
        self.assertAlmostEqual(clean["security"], 1.0)

    def test_fast_backoff_respects_deadline(self):
        c = ArenaClient("http://127.0.0.1:9", "t")
        self.assertFalse(c._sleep_fast(0, time.monotonic() - 1))
        t = time.monotonic()
        self.assertTrue(c._sleep_fast(10, time.monotonic() + 0.4))
        self.assertLess(time.monotonic() - t, 0.4)
        c.close()


if __name__ == "__main__":
    unittest.main()
