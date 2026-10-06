"""
Opponent modelling: who else is in the market this round, and what will they bid?

Every number a strategy needs - the field S_k it is bidding against - is the sum
of the other nodes' bids. The strategy primer recovers it from last round's
clearing price, which is right on average and wrong exactly when it matters:
the field changes shape every time a node runs flat and sits a round out.

This module does better, using only public information:

* `/v1/swarm` publishes every node's device (weights, floors, mobility) and its
  battery to four decimals, so we know who is admissible this round before we
  bid.
* The three baseline bots are open source (`baselines/bot.py`). Each is a pure
  function of public round data and its own last result, so a shadow copy fed
  the same inputs reproduces its bid to the sixth decimal.
* Anything we do not recognise - another team's agent, a changed bot - is
  covered by a residual: the published clearing price tells us the true total
  bid after every round, and whatever our shadows did not explain is tracked
  and forecast from that.

`Market.accuracy` reports how well the shadows have matched the truth so far.
If it degrades, the planner widens its safety margins automatically instead of
trusting a model that has stopped describing the world.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from econ import RESOURCES

Bid = Dict[str, float]


# ---------------------------------------------------------------------------
# The three published baselines, restated. Logic mirrors baselines/bot.py.
# ---------------------------------------------------------------------------
FLOOR_MARGIN = 1.20


def _naive_max(budget, prices, caps, prof, last) -> Bid:
    w = prof["weights"]
    return {k: budget * float(w.get(k, 1 / 3)) for k in RESOURCES}


def _even_split(budget, prices, caps, prof, last) -> Bid:
    return {k: budget / 3.0 for k in RESOURCES}


def _estimate_others(last: Optional[dict]) -> Dict[str, float]:
    """The bot's own (one-round-stale) estimate of the field. Bug-for-bug."""
    if not last:
        return {k: 1.0 for k in RESOURCES}
    return {
        k: max(1e-4, float(last["prices"][k]) * float(last["capacities"][k])
               - float(last["bid"].get(k, 0.0)))
        for k in RESOURCES
    }


def _buy_floors(bid, budget, caps, q_t, s_t, others) -> Bid:
    bid = dict(bid)
    for k, floor in (("compute", q_t), ("security", s_t)):
        cap = float(caps.get(k, 1.0))
        s_k = max(float(others.get(k, 1.0)), 1e-9)
        if floor <= 0.0 or floor >= cap:
            continue
        needed = s_k * floor / (cap - floor)
        if needed <= bid.get(k, 0.0):
            continue
        deficit = needed - bid[k]
        donors = [r for r in RESOURCES if r != k and bid.get(r, 0.0) > 0.0]
        available = sum(bid[r] for r in donors)
        headroom = max(0.0, budget - sum(bid.values()))
        take = min(deficit, headroom + available)
        if take <= 0.0:
            continue
        from_donors = take - min(take, headroom)
        if from_donors > 0.0 and available > 1e-9:
            for r in donors:
                bid[r] -= from_donors * (bid[r] / available)
        bid[k] += take
    total = sum(bid.values())
    if total > budget and total > 1e-9:
        bid = {k: v * (budget / total) for k, v in bid.items()}
    return {k: max(0.0, v) for k, v in bid.items()}


def _proportional(budget, prices, caps, prof, last) -> Bid:
    w = prof["weights"]
    scores = {}
    for k in RESOURCES:
        price = max(float(prices.get(k, 1.0)), 0.05)
        weight = float(w.get(k, 1 / 3))
        scores[k] = weight * weight * float(caps.get(k, 1.0)) / price
    total = sum(scores.values()) or 1.0
    bid = {k: budget * v / total for k, v in scores.items()}
    q_t = min(float(prof["q_min"]) * FLOOR_MARGIN, 0.95 * float(caps.get("compute", 1.0)))
    s_t = min(float(prof["s_min"]) * FLOOR_MARGIN, 0.95 * float(caps.get("security", 1.0)))
    return _buy_floors(bid, budget, caps, q_t, s_t, _estimate_others(last))


SHADOWS: Dict[str, Callable[..., Bid]] = {
    "bot-naive-max": _naive_max,
    "bot-even-split": _even_split,
    "bot-proportional": _proportional,
}


def clamp(bid: Bid, budget: float) -> Bid:
    """What the bots' runner and the arena do to a bid before it counts."""
    clean = {k: max(0.0, float(bid.get(k, 0.0))) for k in RESOURCES}
    total = sum(clean.values())
    if total > budget and total > 0:
        clean = {k: v * (budget / total) for k, v in clean.items()}
    return {k: round(v, 6) for k, v in clean.items()}


# ---------------------------------------------------------------------------
# Device physics
# ---------------------------------------------------------------------------
@dataclass
class Physics:
    """Battery model: ``drain = a * x_E * (1 + mobility) + c``; rest recharges ``r``.

    The arena does not publish a, c or r. The defaults are the published
    scenario values; `fit` re-estimates a and c from our own round results
    (which carry `battery_drawn` and the energy share that caused it), so the
    agent keeps working if an organiser retunes the physics.
    """

    drain: float = 0.30
    idle: float = 0.004
    recharge: float = 0.22
    cutoff: float = 0.05
    fitted: bool = False

    def cost(self, x_energy: float, mobility: float) -> float:
        return self.drain * x_energy * (1.0 + mobility) + self.idle

    def fit(self, history: List[dict], mobility: float) -> None:
        pts = [(float(h["allocation"]["energy"]) * (1.0 + mobility), float(h["battery_drawn"]))
               for h in history[-30:]
               if "battery_drawn" in h and "allocation" in h]
        if len(pts) < 3:
            return
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        if sxx < 1e-6:
            # Every round drew the same share: only the intercept is identified.
            self.idle = max(0.0, my - self.drain * mx)
            return
        slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
        if 0.05 < slope < 2.0:
            self.drain = slope
            self.idle = max(0.0, my - slope * mx)
            self.fitted = True

    def observe_rest(self, before: float, after: float) -> None:
        """A node that sat a round out recharged from `before` to `after`."""
        gained = after - before
        if after < 0.999 and 0.02 < gained < 0.8:
            self.recharge = 0.8 * self.recharge + 0.2 * gained


# ---------------------------------------------------------------------------
# The market
# ---------------------------------------------------------------------------
@dataclass
class Node:
    node_id: str
    team: str
    shadow: Optional[Callable[..., Bid]]
    weights: Dict[str, float]
    q_min: float
    s_min: float
    mobility: float
    battery: float
    active: bool = True
    ejected: bool = False
    last_record: Optional[dict] = None     # what the bot's own history[-1] holds
    predicted: Optional[Bid] = None        # our forecast of its bid this round

    def profile(self) -> Dict[str, Any]:
        return {"weights": self.weights, "q_min": self.q_min, "s_min": self.s_min}

    def admissible(self, cutoff: float) -> bool:
        return self.active and not self.ejected and self.battery > cutoff + 1e-12


@dataclass
class Market:
    """Everyone except us, as best we can see them."""

    me: Optional[str] = None
    physics: Physics = field(default_factory=Physics)
    nodes: Dict[str, Node] = field(default_factory=dict)
    rounds: Dict[int, dict] = field(default_factory=dict)   # round -> public payload
    my_bids: Dict[int, Bid] = field(default_factory=dict)
    predictions: Dict[int, Dict[str, Bid]] = field(default_factory=dict)
    residual: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in RESOURCES})
    errors: List[float] = field(default_factory=list)

    # -------------------------------------------------------------- observe
    def observe_swarm(self, rows: List[dict]) -> None:
        """Sync devices and live batteries from `/v1/swarm`."""
        for row in rows or []:
            nid = row.get("node_id")
            if not nid or nid == self.me:
                continue
            feats = row.get("features") or {}
            node = self.nodes.get(nid)
            if node is None:
                node = Node(
                    node_id=nid,
                    team=row.get("team", ""),
                    shadow=SHADOWS.get(row.get("team", "")),
                    weights=dict(row.get("weights") or {}),
                    q_min=float(row.get("q_min", 0.0)),
                    s_min=float(row.get("s_min", 0.0)),
                    mobility=float(feats.get("mobility", 0.0)),
                    battery=float(feats.get("battery", 1.0)),
                )
                self.nodes[nid] = node
            # Devices can be re-cloned when a participant registers, so the
            # profile is refreshed every time rather than read once.
            node.weights = dict(row.get("weights") or node.weights)
            node.q_min = float(row.get("q_min", node.q_min))
            node.s_min = float(row.get("s_min", node.s_min))
            node.mobility = float(feats.get("mobility", node.mobility))
            node.battery = float(feats.get("battery", node.battery))
            node.active = bool(row.get("active", True))
            node.ejected = bool(row.get("ejected", False))

    def observe_round(self, payload: dict) -> None:
        """Record a round's public data and learn from the round before it.

        The prices published with round t are the prices that cleared round
        t-1, so this is the moment round t-1's true total bid becomes known.
        """
        r = int(payload["round"])
        if r in self.rounds:
            return
        self.rounds[r] = {
            "round": r,
            "prices": dict(payload.get("prices") or {}),
            "capacities": dict(payload.get("capacities") or {}),
            "budget": payload.get("budget"),
        }
        prev = self.rounds.get(r - 1)
        if prev is not None:
            self._reconcile(prev, self.rounds[r]["prices"])

    def _reconcile(self, prev: dict, cleared: Dict[str, float]) -> None:
        """Compare round t-1's forecast with the truth the clearing price reveals."""
        r = prev["round"]
        preds = self.predictions.get(r)
        if preds is None:
            return
        mine = self.my_bids.get(r, {})
        gaps: Dict[str, float] = {}
        for k in RESOURCES:
            true_total = float(cleared.get(k, 0.0)) * float(prev["capacities"].get(k, 1.0))
            modelled = float(mine.get(k, 0.0)) + sum(b.get(k, 0.0) for b in preds.values())
            if true_total <= 0.0101 * float(prev["capacities"].get(k, 1.0)):
                continue  # the published price is floored at 0.01: uninformative
            gap = true_total - modelled
            gaps[k] = gap
            self.errors.append(abs(gap) / max(true_total, 1e-6))
            # Whatever the shadows did not explain is somebody we cannot see.
            self.residual[k] = 0.6 * self.residual[k] + 0.4 * max(0.0, gap)
        self.errors = self.errors[-60:]

        # Self-healing. `proportional` is the only bot whose bid depends on its
        # own past, so a forecast that went wrong once (a round we never saw,
        # say) would otherwise stay wrong. If it is the only node in the round
        # we could not pin down exactly, the clearing price tells us what it
        # really bid, and its history is corrected from that.
        stateful = [nid for nid in preds if self.nodes[nid].shadow is _proportional]
        strangers = any(n.shadow is None and n.admissible(self.physics.cutoff)
                        for n in self.nodes.values())
        if len(stateful) == 1 and not strangers and gaps:
            nid = stateful[0]
            preds[nid] = {k: round(max(0.0, preds[nid][k] + gaps.get(k, 0.0)), 6)
                          for k in RESOURCES}

        # Advance each bot's private history exactly as its own client would.
        for nid, bid in preds.items():
            node = self.nodes.get(nid)
            if node is not None and node.shadow is not None:
                node.last_record = {
                    "prices": prev["prices"],
                    "capacities": prev["capacities"],
                    "bid": bid,
                }

    # -------------------------------------------------------------- predict
    def predict(self, round_index: int) -> Dict[str, float]:
        """The field S_k this round: everyone else's total bid on each pool."""
        pub = self.rounds[round_index]
        budget = float(pub.get("budget") or 1.0)
        preds: Dict[str, Bid] = {}
        unexplained = False
        for nid, node in self.nodes.items():
            node.predicted = None
            if not node.admissible(self.physics.cutoff):
                continue
            if node.shadow is None:
                unexplained = True
                continue
            raw = node.shadow(budget, pub["prices"], pub["capacities"],
                              node.profile(), node.last_record)
            node.predicted = clamp(raw, budget)
            preds[nid] = node.predicted
        self.predictions[round_index] = preds
        field_ = {k: sum(b[k] for b in preds.values()) for k in RESOURCES}
        if unexplained or self.accuracy > 0.02:
            for k in RESOURCES:
                field_[k] += self.residual[k]
        return field_

    def record_my_bid(self, round_index: int, bid: Bid) -> None:
        self.my_bids[round_index] = dict(bid)

    @property
    def accuracy(self) -> float:
        """Mean relative error of our field forecast over recent rounds."""
        if not self.errors:
            return 0.0
        recent = self.errors[-12:]
        return sum(recent) / len(recent)

    def awake(self) -> List[str]:
        return [n.team for n in self.nodes.values() if n.admissible(self.physics.cutoff)]
