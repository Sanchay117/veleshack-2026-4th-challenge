"""
The agent main loop.

Built on the template's loop, which already registers, keeps its lease alive
on its own thread, bids before doing any bookkeeping, and survives the faults
the graded arena throws at it. Three additions:

* Every round, before deciding, it reads `/v1/swarm` - who is in the swarm and
  how much charge each node has - and feeds it to the strategist together
  with the round. It does this on rounds it sits out too, so the opponent
  model never loses track. `/v1/swarm` is exempt from fault injection; if it
  fails anyway the strategist falls back to estimating the field from prices.
* The decision itself goes to `strategy.Strategist` with the run length and
  round number, which the template signature does not carry.
* One log line per round says what was decided and why, and an optional
  telemetry server (`TELEMETRY_PORT`) feeds the live dashboard.

Run it:
    ARENA_URL=http://localhost:8080 TEAM_NAME=team-kappa python agent.py

Configuration is environment only:
    ARENA_URL        arena base URL                    (default http://localhost:8080)
    TEAM_NAME        your team; decides your device     (default unnamed-team)
    LOG_LEVEL        DEBUG / INFO / WARNING             (default INFO)
    TELEMETRY_PORT   serve the dashboard on this port   (default off)
    TELEMETRY_LINGER seconds to keep it up after the run (default 600; telemetry only)

Copyright 2026 The CoGNETs Consortium, Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from typing import Any, Dict, List, Optional

from client import (
    AlreadyBid,
    ArenaClient,
    ArenaClientError,
    Ejected,
    NoRound,
    NotAdmissible,
    RoundClosed,
    WrongRound,
)
from strategy import BRAIN, safe_bid
from telemetry import Telemetry, serve

LOG = logging.getLogger("agent")

ARENA_URL = os.environ.get("ARENA_URL", "http://localhost:8080")
TEAM_NAME = os.environ.get("TEAM_NAME", "unnamed-team")
TELEMETRY_PORT = os.environ.get("TELEMETRY_PORT", "").strip()

_stop = threading.Event()


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------
def heartbeat_loop(client: ArenaClient, interval: float) -> None:
    """Renew the lease on its own schedule.

    Deliberately a separate thread. If you heartbeat only inside the bidding
    loop, then one slow round - or one round you sit out because your battery
    is flat - lets your lease expire, and you silently stop being scored.
    """
    LOG.info("heartbeat every %.1fs", interval)
    while not _stop.is_set():
        try:
            client.heartbeat()
        except Ejected:
            LOG.error("ejected from the run; heartbeat thread stopping")
            return
        except ArenaClientError as exc:
            LOG.warning("heartbeat failed: %s", exc)
        _stop.wait(interval)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run() -> int:
    client = ArenaClient(ARENA_URL, TEAM_NAME)
    telemetry: Optional[Telemetry] = None
    if TELEMETRY_PORT:
        telemetry = Telemetry(TEAM_NAME)
        serve(telemetry, int(TELEMETRY_PORT))

    LOG.info("connecting to %s as '%s'", ARENA_URL, TEAM_NAME)
    if not client.wait_for_arena(timeout=90.0):
        LOG.error("arena at %s never became healthy", ARENA_URL)
        return 1

    client.register()

    lease = float(client.arena_info.get("lease_seconds", 12.0))
    total_rounds = int(client.arena_info.get("total_rounds", 0))
    cutoff = client.arena_info.get("battery_cutoff")
    if cutoff is not None:
        BRAIN.market.physics.cutoff = float(cutoff)
    hb = threading.Thread(
        target=heartbeat_loop, args=(client, max(1.0, lease / 3.0)), daemon=True
    )
    hb.start()

    history: List[Dict[str, Any]] = []
    owed: List[int] = []          # rounds we bid in whose results we have not read
    last_bid_round = 0
    last_seen_round = 0
    idle_polls = 0

    while not _stop.is_set():
        # ---------------------------------------------------------- read round
        try:
            rnd = client.get_round()
        except NoRound:
            idle_polls += 1
            if idle_polls % 20 == 1:
                LOG.info("no round open yet, waiting")
            _stop.wait(0.5)
            continue
        except Ejected as exc:
            LOG.error("ejected: %s", exc)
            return 2
        except ArenaClientError as exc:
            LOG.warning("could not read the round: %s", exc)
            _stop.wait(1.0)
            continue

        idle_polls = 0
        round_index = int(rnd["round"])

        # ------------------------------------------------------ fresh arena?
        # A restarted arena counts from round 1 again. Everything keyed off
        # the round number, the history, and the strategist's model of the
        # swarm describe a run that no longer exists, so all of it is dropped.
        if round_index < max(last_bid_round, last_seen_round):
            LOG.info(
                "round counter went backwards (%d -> %d): a new run has "
                "started on this arena, resetting per-run state",
                max(last_bid_round, last_seen_round),
                round_index,
            )
            last_bid_round = last_seen_round = 0
            history.clear()
            owed.clear()
            BRAIN.reset()
            if cutoff is not None:
                BRAIN.market.physics.cutoff = float(cutoff)
            if telemetry:
                telemetry.reset()

        # -------------------------------------------------- observe the swarm
        # Once per round, as early as possible: who is awake, how much charge
        # each node has. Cheap (one fault-exempt GET) and done even on rounds
        # we sit out, so the opponent model never loses the thread.
        if round_index != last_seen_round and not rnd.get("settled"):
            last_seen_round = round_index
            _observe(client, rnd, telemetry)

        # ----------------------------------------------------- already done?
        # Bidding comes BEFORE fetching results. Reading results is
        # bookkeeping; missing the bidding window is a lost round.
        #
        # The template asked only for the result of the round it had just bid
        # on - which never settles while that round is open, and by the time
        # it has, the next round is open and the loop bids instead. So it
        # collected almost nothing (1 result in 49 rounds, measured). Results
        # are now owed per round and collected once that round is over.
        if round_index <= last_bid_round or rnd.get("settled"):
            _collect_owed(client, history, owed, round_index, bool(rnd.get("settled")))
            _stop.wait(min(0.3, max(0.05, float(rnd.get("seconds_remaining", 0.3)))))
            continue

        you = rnd.get("you") or {}
        if you.get("already_submitted"):
            last_bid_round = round_index
            continue
        if you.get("admissible") is False:
            LOG.info("round %d: resting (battery %.3f), sitting it out",
                     round_index, float(you.get("battery", 0.0)))
            BRAIN.note_rest(rnd)
            if telemetry:
                telemetry.on_decision(BRAIN.last)
            last_bid_round = round_index
            _stop.wait(0.4)
            continue

        # ------------------------------------------------------------- decide
        budget = float(rnd.get("budget", 0.0))
        try:
            bid = BRAIN.decide(rnd, client.profile, history, total_rounds)
            _log_decision(BRAIN.last)
            if telemetry:
                telemetry.on_decision(BRAIN.last)
        except Exception:
            # A crash in the strategy must never kill the agent. The safe bid
            # costs a little score; crashing costs the whole run.
            LOG.exception("strategy raised; falling back to the safe bid")
            bid = safe_bid(budget, client.profile)

        bid = _sanitise(bid, budget)

        # ---------------------------------------------------------------- bid
        try:
            ack = client.post_bid(round_index, bid, deadline=rnd.get("_deadline"))
            last_bid_round = round_index
            owed.append(round_index)
            if ack.get("warnings"):
                LOG.warning("round %d accepted with warnings: %s",
                            round_index, ack["warnings"])
        except AlreadyBid:
            last_bid_round = round_index
        except (RoundClosed, WrongRound) as exc:
            LOG.info("round %d: %s", round_index, exc)
            last_bid_round = round_index
        except NotAdmissible as exc:
            LOG.info("round %d: %s", round_index, exc)
            last_bid_round = round_index
        except Ejected as exc:
            LOG.error("ejected: %s", exc)
            return 2
        except ArenaClientError as exc:
            LOG.warning("round %d: bid failed: %s", round_index, exc)

        # ------------------------------------------------------------- finish
        if total_rounds and round_index >= total_rounds:
            LOG.info("final round submitted; waiting for it to settle")
            _stop.wait(3.0)
            try:
                final = client.me()
                LOG.info(
                    "FINAL  score=%.3f  rounds=%d  idle=%d  missed=%d  floors=%d  kappa=%.2f",
                    final.get("score", 0.0),
                    final.get("rounds_participated", 0),
                    final.get("rounds_idle", 0),
                    final.get("rounds_missed", 0),
                    final.get("floor_violations", 0),
                    final.get("compromise", 0.0),
                )
            except ArenaClientError:
                pass
            if telemetry:
                _observe_board(client, total_rounds + 1, telemetry)
                # Keep the dashboard up so the final standings can be seen.
                # Only with telemetry on: a graded agent exits straight away.
                linger = float(os.environ.get("TELEMETRY_LINGER", "600"))
                LOG.info("run over; dashboard stays up for %.0fs (Ctrl-C to quit)", linger)
                _stop.wait(linger)
            break

        _stop.wait(0.25)

    client.close()
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _observe(client: ArenaClient, rnd: Dict[str, Any],
             telemetry: Optional[Telemetry]) -> None:
    """Snapshot the swarm for this round and hand it to the strategist."""
    swarm = None
    try:
        swarm = client.swarm().get("nodes")
    except ArenaClientError as exc:
        LOG.warning("round %s: no swarm snapshot (%s); estimating from prices",
                    rnd.get("round"), exc)
    try:
        BRAIN.observe(rnd, swarm, client.node_id)
    except Exception:
        LOG.exception("strategist could not digest round %s", rnd.get("round"))
    if telemetry:
        _observe_board(client, int(rnd["round"]), telemetry)


def _observe_board(client: ArenaClient, round_index: int, telemetry: Telemetry) -> None:
    try:
        telemetry.on_board(round_index, client.leaderboard())
    except ArenaClientError:
        pass


def _log_decision(d: Dict[str, Any]) -> None:
    bid = d.get("bid", {})
    LOG.info(
        "round %d  battery=%.3f  awake=%d  bid=C%.3f/E%.3f/S%.3f  x_E=%.3f  "
        "exp_u=%.3f  shadow=%.2f  model=%s(err %.4f)  %.0fms",
        d.get("round", 0), d.get("battery", 0.0), len(d.get("awake", [])),
        bid.get("compute", 0.0), bid.get("energy", 0.0), bid.get("security", 0.0),
        d.get("x_energy", 0.0), d.get("expected_utility", 0.0),
        d.get("shadow_price", 0.0), d.get("model", "?"), d.get("model_error", 0.0),
        d.get("think_ms", 0.0),
    )


def _collect_owed(client: ArenaClient, history: List[Dict[str, Any]], owed: List[int],
                  current: int, current_settled: bool) -> None:
    """Read the results of rounds that are over. Best-effort, one per poll.

    A round is over once a later round is open (or the final round reports
    itself settled). Rounds that stay unreadable for a few rounds are given up
    on, so a long fault burst cannot build a backlog in front of the bid.
    """
    owed[:] = [r for r in owed if r >= current - 4]
    ready = [r for r in owed if r < current or (r == current and current_settled)]
    if not ready:
        return
    r = ready[0]
    try:
        result = client.get_result(r)
    except NoRound:
        return
    except ArenaClientError:
        return
    owed.remove(r)
    if not result.get("participated"):
        return
    if any(h.get("round") == result.get("round") for h in history[-5:]):
        return
    history.append(result)
    history.sort(key=lambda h: int(h.get("round", 0)))
    LOG.info(
        "round %d  result  utility=%.4f%s  battery=%.3f  total=%.3f",
        result["round"],
        result.get("utility", 0.0),
        _fmt_violations(result.get("floor_violations")),
        result.get("battery", 0.0),
        result.get("cumulative_score", 0.0),
    )


def _sanitise(bid: Any, budget: float) -> Dict[str, float]:
    """Last line of defence before a bid leaves the process.

    The arena raises your compromise score for negative, non-numeric or
    over-budget bids, and ejects you if it happens often enough. Clamping here
    means a strategy bug costs score rather than the run.
    """
    resources = ("compute", "energy", "security")
    clean: Dict[str, float] = {}
    for k in resources:
        try:
            v = float((bid or {}).get(k, 0.0))
        except (TypeError, ValueError, AttributeError):
            v = 0.0
        if v != v or v in (float("inf"), float("-inf")) or v < 0.0:
            v = 0.0
        clean[k] = v
    total = sum(clean.values())
    if total > budget and total > 0:
        scale = budget / total
        clean = {k: v * scale for k, v in clean.items()}
    return clean


def _fmt_violations(violations: Optional[List[str]]) -> str:
    return f"  MISSED:{','.join(violations)}" if violations else ""


def _handle_signal(signum: int, _frame: object) -> None:
    LOG.info("signal %d received, shutting down", signum)
    _stop.set()


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # One line per request drowns the decision log; keep it for DEBUG only.
    if logging.getLogger().level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    finally:
        _stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
