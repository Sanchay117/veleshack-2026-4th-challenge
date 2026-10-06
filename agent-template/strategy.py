"""
Strategy: forecast the swarm, plan the battery, buy the floors exactly.

The whole decision, every round:

1. **See the field.** `opponents.Market` knows who is awake this round (from
   `/v1/swarm`) and reproduces the baseline bots' bids exactly, with a
   residual for anything it cannot explain. So S_k is known, not guessed.
2. **Forecast the rest of the run.** The bots never think about their
   batteries, so their naps are predictable: roll their shadows forward and
   we know, rounds ahead, when the pools will be quiet.
3. **Plan.** `planner.plan` solves a dynamic programme over our battery
   for the remaining rounds and returns this round's best energy spend, with
   the compute/security split that buys both floors at the exact minimum.
4. Iterate 2-3 because our own energy bid changes how fast the bots drain.

Fallbacks are layered so a surprise costs score, never the run: no swarm data
means an empirical field estimate from the clearing price; a model that stops
matching the truth widens its floor margins; any exception at all drops to a
battery-aware proportional bid.

`decide_bid` keeps the template's signature so this file is still a drop-in
for `agent.py`, `runner.py` or the organisers' harness.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from econ import RESOURCES
from opponents import Market, clamp
from planner import (ENERGY_FRACTIONS, Device, Market1, Menu, Plan, plan, round_menu,
                     value)

LOG = logging.getLogger("agent.strategy")

#: Rollout/plan passes. Our energy bid changes how fast the bots drain, which
#: changes the forecast the plan was built on; two or three passes settle it.
PLAN_ITERATIONS = 3

#: The default energy spend assumed for ourselves before a plan exists.
DEFAULT_ACTION = int(np.searchsorted(ENERGY_FRACTIONS, 0.03))

#: Seconds of thinking after which optional refinement passes are skipped.
#: The graded round is 4 s and the bid may meet 250 ms of latency plus retries,
#: so the first full plan (~50-150 ms) is always made and extras must fit.
THINK_BUDGET = 0.6

#: Spread of the capacity draws used when planning over sampled futures
#: (the published scenarios draw each pool uniformly within +-30%).
CAPACITY_JITTER = 0.30

#: Value of one unit of charge when the planner is switched off (ablation only).
FIXED_SHADOW_PRICE = 1.0

#: How far above a service floor to aim. With the field known exactly the only
#: error left is the arena's six-decimal rounding, so a sliver is enough; the
#: margin grows with the model's measured error. Without the model the field
#: is a one-round-stale estimate and needs the bots' 20%.
MIN_FLOOR_MARGIN = 0.003
EMPIRICAL_FLOOR_MARGIN = 0.20


class Strategist:
    """Stateful decision-maker. One per agent process, reset per run."""

    def __init__(self, use_opponent_model: bool = True, use_planner: bool = True,
                 lookahead: int = 0, scenarios: int = 1) -> None:
        self.use_opponent_model = use_opponent_model
        self.use_planner = use_planner
        self.lookahead = lookahead
        self.scenarios = scenarios
        self.reset()

    def reset(self) -> None:
        self.market = Market()
        self.plan: Optional[Plan] = None
        self.cap_sum = {k: 5.0 for k in RESOURCES}     # prior: five rounds at 1.0
        self.budget_sum, self.n_obs = 5.0, 5
        self.last: Dict[str, Any] = {}
        self.fields: Dict[int, Dict[str, float]] = {}   # round -> modelled S_k
        self.my_rest_from: Optional[float] = None

    # ------------------------------------------------------------- observe
    def observe(self, payload: Dict[str, Any], swarm: Optional[List[dict]] = None,
                me: Optional[str] = None) -> None:
        """Feed every round the agent sees, including rounds it sits out.

        The field is forecast here rather than in `decide`, so the bots'
        shadow histories keep advancing through rounds we spend resting.
        `swarm` must be fresh for this round (fetched after it opened); without
        it we do not know who is awake, and `decide` falls back to prices.
        """
        if me:
            self.market.me = me
        r = int(payload.get("round", 0))
        if not r:
            return
        if r not in self.market.rounds:
            self.market.observe_round(payload)
            caps = payload.get("capacities") or {}
            for k in RESOURCES:
                self.cap_sum[k] += float(caps.get(k, 1.0))
            self.budget_sum += float(payload.get("budget") or 1.0)
            self.n_obs += 1
        if swarm is not None and self.use_opponent_model and r not in self.fields:
            self.market.observe_swarm(swarm)
            self.fields[r] = self.market.predict(r)
            for old in [k for k in self.fields if k < r - 5]:
                del self.fields[old]
        you = payload.get("you") or {}
        if "battery" in you:
            b = float(you["battery"])
            if you.get("admissible") is False:
                self.my_rest_from = b
            elif self.my_rest_from is not None:
                self.market.physics.observe_rest(self.my_rest_from, b)
                self.my_rest_from = None

    # -------------------------------------------------------------- decide
    def decide(self, payload: Dict[str, Any], profile: Dict[str, Any],
               history: List[Dict[str, Any]], total_rounds: int) -> Dict[str, float]:
        t0 = time.perf_counter()
        r = int(payload["round"])
        budget = float(payload.get("budget") or 0.0)
        given = payload.get("capacities") or {}
        caps = {k: float(given.get(k, 1.0)) for k in RESOURCES}
        feats = profile.get("features") or {}
        dev = Device(
            weights={k: float(v) for k, v in profile["weights"].items()},
            q_min=float(profile.get("q_min", 0.0)),
            s_min=float(profile.get("s_min", 0.0)),
            mobility=float(feats.get("mobility", 0.0)),
        )
        battery = float((payload.get("you") or {}).get("battery", feats.get("battery", 1.0)))
        phys = self.market.physics
        phys.fit(history, dev.mobility)

        modelled = r in self.fields
        field_now = self.fields[r] if modelled else empirical_field(history, payload)
        margin = (min(0.25, max(MIN_FLOOR_MARGIN, 4.0 * self.market.accuracy)) if modelled
                  else EMPIRICAL_FLOOR_MARGIN)

        now = Market1(budget=budget, capacities=caps, field=field_now)
        menu_now = round_menu(now, dev, phys.drain, phys.idle, margin)

        horizon = max(1, int(total_rounds) - r + 1) if total_rounds else 1
        forecast_awake: List[List[str]] = []
        if self.use_planner and horizon > 1:
            actions = self._prior_actions(r, horizon)
            for i in range(PLAN_ITERATIONS):
                if i and time.perf_counter() - t0 > THINK_BUDGET:
                    break   # a good plan now beats a perfect one after the round closes
                futures, forecast_awake = (
                    self._forecast(r, horizon, actions, dev, battery, now) if modelled
                    else ([now] * (horizon - 1), []))
                menus = [menu_now] + [round_menu(m, dev, phys.drain, phys.idle, margin)
                                      for m in futures]
                p = plan(battery, menus, phys.cutoff, phys.recharge)
                if p.action_path == actions:
                    break
                actions = p.action_path
            if self.scenarios > 1 and modelled and time.perf_counter() - t0 < THINK_BUDGET / 2:
                # Plan against the average of several sampled futures instead of
                # the single mean one: the bots' naps shift with the pool sizes.
                rng = np.random.default_rng(r)
                runs = []
                for _ in range(self.scenarios):
                    fut, _aw = self._forecast(r, horizon, p.action_path, dev, battery, now, rng)
                    runs.append([round_menu(m, dev, phys.drain, phys.idle, margin) for m in fut])
                avg = [Menu(utility=np.mean([run[t].utility for run in runs], axis=0),
                            drain=np.mean([run[t].drain for run in runs], axis=0),
                            bids=runs[0][t].bids,
                            x_energy=np.mean([run[t].x_energy for run in runs], axis=0))
                       for t in range(horizon - 1)]
                p = plan(battery, [menu_now] + avg, phys.cutoff, phys.recharge)
            a = p.action
            if self.lookahead and modelled:
                a = self._lookahead(r, horizon, p, menu_now, dev, battery, now, margin)
            self.plan = p
            self.plan_round = r
        else:
            # The last round: leftover charge is worth nothing, so spend it.
            # With planning switched off (an ablation) charge gets a fixed price.
            q = menu_now.utility
            if not self.use_planner and horizon > 1:
                q = q - FIXED_SHADOW_PRICE * menu_now.drain
            a = int(np.argmax(q))
            self.plan = None

        bid = dict(zip(("compute", "energy", "security"), map(float, menu_now.bids[a])))
        bid = clamp(bid, budget, down=True)
        if modelled:
            self.market.record_my_bid(r, bid)

        self.last = {
            "round": r,
            "resting": False,
            "battery": battery,
            "budget": budget,
            "bid": bid,
            "field": {k: round(v, 5) for k, v in field_now.items()},
            "capacities": caps,
            "predicted_bids": ({n.team: n.predicted for n in self.market.nodes.values()
                                if n.predicted} if modelled else {}),
            "awake": self.market.awake(),
            "energy_fraction": float(ENERGY_FRACTIONS[a]),
            "x_energy": float(menu_now.x_energy[a]),
            "expected_utility": float(menu_now.utility[a]),
            "drain": float(menu_now.drain[a]),
            "margin": margin,
            "model": "shadow" if modelled else "empirical",
            "model_error": self.market.accuracy,
            "plan_battery": self.plan.battery_path if self.plan else [battery],
            "plan_actions": self.plan.action_path if self.plan else [a],
            "forecast_awake": forecast_awake,
            "shadow_price": self.plan.shadow_price if self.plan else 0.0,
            "physics": {"drain": phys.drain, "idle": phys.idle,
                        "recharge": phys.recharge, "fitted": phys.fitted},
            "think_ms": round(1000 * (time.perf_counter() - t0), 1),
        }
        return bid

    def note_rest(self, payload: Dict[str, Any]) -> None:
        """Keep `last` current through a round we sit out on a flat battery.

        Nothing is decided, but logs and the dashboard should still show the
        round, who is awake, and the plan shifted forward by the rounds passed.
        """
        r = int(payload.get("round", 0))
        prev = self.last or {}
        s = max(0, r - int(prev.get("round", r)))
        you = payload.get("you") or {}
        self.last = {
            **prev,
            "round": r,
            "resting": True,
            "battery": float(you.get("battery", prev.get("battery", 0.0))),
            "bid": {k: 0.0 for k in RESOURCES},
            "energy_fraction": 0.0, "x_energy": 0.0, "expected_utility": 0.0, "drain": 0.0,
            "capacities": dict(payload.get("capacities") or {}),
            "predicted_bids": {n.team: n.predicted for n in self.market.nodes.values()
                               if n.predicted} if r in self.fields else {},
            "awake": self.market.awake(),
            "plan_battery": list(prev.get("plan_battery") or [])[s:],
            "plan_actions": list(prev.get("plan_actions") or [])[s:],
            "forecast_awake": list(prev.get("forecast_awake") or [])[s:],
            "think_ms": 0.0,
        }

    # ----------------------------------------------------------- internals
    def _lookahead(self, r: int, horizon: int, p: Plan, menu_now, dev: Device,
                   battery: float, now: Market1, margin: float) -> int:
        """Re-score the best few actions with the swarm re-simulated for each.

        The DP treats the forecast as fixed, but our energy bid this round
        changes how much energy the bots win, how fast they drain, and so when
        they nap. For each short-listed action, roll the swarm forward again
        under it and value the rest of the run on that forecast.
        """
        phys = self.market.physics
        short = [int(i) for i in np.argsort(-p.q_values)[: self.lookahead]]
        best_a, best_v = p.action, -np.inf
        for c in short:
            acts = [c] + list(p.action_path[1:])
            futures, _ = self._forecast(r, horizon, acts, dev, battery, now)
            menus = [round_menu(m, dev, phys.drain, phys.idle, margin) for m in futures]
            after = max(0.0, battery - float(menu_now.drain[c]))
            v = float(menu_now.utility[c]) + value(after, menus, phys.cutoff, phys.recharge)
            if v > best_v + 1e-9:
                best_a, best_v = c, v
        return best_a

    def _prior_actions(self, r: int, horizon: int) -> List[int]:
        """Last round's plan, shifted to start at this round."""
        if self.plan is not None and getattr(self, "plan_round", None) == r - 1:
            prior = list(self.plan.action_path[1:])
        else:
            prior = []
        prior = prior[:horizon]
        return prior + [DEFAULT_ACTION] * (horizon - len(prior))

    def _forecast(self, r: int, horizon: int, actions: List[int], dev: Device,
                  battery: float, now: Market1, rng: Optional[np.random.Generator] = None):
        """Roll the swarm forward under our planned actions.

        Returns one `Market1` per future round, and who is expected awake in
        each. Future capacities are their running means, or with `rng` a
        draw around them; the bots' bids, batteries and private histories are
        simulated exactly.
        """
        mk = self.market
        phys = mk.physics
        n = self.n_obs
        mean_caps = {k: self.cap_sum[k] / n for k in RESOURCES}
        mean_budget = self.budget_sum / n
        jitter = CAPACITY_JITTER

        bots = [
            {"node": nd, "battery": nd.battery, "last": nd.last_record,
             "alive": nd.active and not nd.ejected}
            for nd in mk.nodes.values() if nd.shadow is not None
        ]
        stranger = {k: (mk.residual[k] if any(nd.shadow is None and nd.admissible(phys.cutoff)
                                              for nd in mk.nodes.values()) else 0.0)
                    for k in RESOURCES}
        w_cs = dev.weights["compute"] / max(dev.weights["compute"] + dev.weights["security"], 1e-9)

        pub_prices = dict(mk.rounds[r]["prices"])
        caps, budget = dict(now.capacities), now.budget
        my_b = battery
        futures: List[Market1] = []
        awake_seq: List[List[str]] = []

        for t in range(horizon):
            # Everyone's bid this round.
            bids: Dict[int, Dict[str, float]] = {}
            awake = []
            for i, bot in enumerate(bots):
                nd = bot["node"]
                if bot["alive"] and bot["battery"] > phys.cutoff + 1e-12:
                    raw = nd.shadow(budget, pub_prices, caps, nd.profile(), bot["last"])
                    bids[i] = clamp(raw, budget)
                    awake.append(nd.team)
            field_ = {k: sum(b[k] for b in bids.values()) + stranger[k] for k in RESOURCES}
            if t > 0:
                futures.append(Market1(budget=budget, capacities=dict(caps), field=field_))
                awake_seq.append(awake)

            # Our own planned bid, so the bots see the competition we will bring.
            a = actions[t] if t < len(actions) else DEFAULT_ACTION
            if my_b <= phys.cutoff + 1e-12 or a < 0:
                mine = {k: 0.0 for k in RESOURCES}
                my_b = min(1.0, my_b + phys.recharge)
            else:
                e = float(ENERGY_FRACTIONS[a]) * budget
                rest = budget - e
                mine = {"compute": rest * w_cs, "energy": e, "security": rest * (1 - w_cs)}
                tot_e = field_["energy"] + e
                x_e = caps["energy"] * e / tot_e if tot_e > 0 else 0.0
                my_b = max(0.0, my_b - phys.cost(x_e, dev.mobility))

            # Settle: bots drain or rest, and prices clear for next round.
            totals = {k: field_[k] + mine[k] for k in RESOURCES}
            for i, bot in enumerate(bots):
                nd = bot["node"]
                if i in bids:
                    x_e = (caps["energy"] * bids[i]["energy"] / totals["energy"]
                           if totals["energy"] > 0 else 0.0)
                    bot["battery"] = max(0.0, bot["battery"] - phys.cost(x_e, nd.mobility))
                    bot["last"] = {"prices": pub_prices, "capacities": dict(caps),
                                   "bid": bids[i]}
                elif bot["alive"]:
                    bot["battery"] = min(1.0, bot["battery"] + phys.recharge)
            pub_prices = {k: max(0.01, totals[k] / max(caps[k], 1e-9)) for k in RESOURCES}
            if rng is None:
                caps = dict(mean_caps)
            else:
                caps = {k: mean_caps[k] * float(rng.uniform(1 - jitter, 1 + jitter))
                        for k in RESOURCES}
            budget = mean_budget

        return futures, awake_seq


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------
def empirical_field(history: List[Dict[str, Any]], payload: Dict[str, Any]) -> Dict[str, float]:
    """The primer's estimator, smoothed: S_k from the price that just cleared.

    The prices published with this round cleared the previous one. If we bid in
    the previous round we know its capacities and our own bid, so S_k is exact
    for that round; otherwise fall back to this round's capacities.
    """
    prices = payload.get("prices") or {}
    caps = payload.get("capacities") or {}
    last = history[-1] if history else None
    prev_round = int(payload.get("round", 0)) - 1
    out = {}
    for k in RESOURCES:
        if last is not None and int(last.get("round", -1)) == prev_round:
            total = float(prices.get(k, 1.0)) * float(last["capacities"][k])
            out[k] = max(1e-4, total - float(last["bid"].get(k, 0.0)))
        else:
            out[k] = max(1e-4, float(prices.get(k, 1.0)) * float(caps.get(k, 1.0)))
    return out


def safe_bid(budget: float, profile: Dict[str, Any]) -> Dict[str, float]:
    """Battery-aware proportional bid. Used only if the strategist raises."""
    w = profile.get("weights", {})
    battery = float((profile.get("features") or {}).get("battery", 1.0))
    keep = min(1.0, max(0.0, (battery - 0.1) / 0.9)) * 0.5
    bid = {k: budget * float(w.get(k, 1 / 3)) for k in RESOURCES}
    freed = bid["energy"] * (1 - keep)
    bid["energy"] -= freed
    bid["compute"] += freed / 2
    bid["security"] += freed / 2
    return clamp(bid, budget, down=True)


# ---------------------------------------------------------------------------
# Template-compatible entry point
# ---------------------------------------------------------------------------
BRAIN = Strategist()


def decide_bid(
    budget: float,
    prices: Dict[str, float],
    capacities: Dict[str, float],
    profile: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Dict[str, float]:
    """The template's signature, for any runner that only knows this function.

    Without the agent's extra context (round number, run length, swarm) the
    strategist still plans, assuming the field it can infer from prices.
    """
    try:
        r = int(history[-1]["round"]) + 1 if history else 1
        payload = {"round": r, "budget": budget, "prices": prices,
                   "capacities": capacities,
                   "you": {"battery": (profile.get("features") or {}).get("battery", 1.0)}}
        BRAIN.observe(payload)
        return BRAIN.decide(payload, profile, history, total_rounds=0)
    except Exception:
        LOG.exception("strategist failed; using the safe bid")
        return safe_bid(budget, profile)
