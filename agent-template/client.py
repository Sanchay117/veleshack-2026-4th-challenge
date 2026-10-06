"""
HTTP client for the CoGNETs Swarm Arena.

THIS FILE IS GIVEN TO YOU AND IT WORKS. You should not need to change it.
Read it, by all means - it is a decent map of the API - but the challenge is
in strategy.py.

What it handles for you:
  * retries with exponential backoff and jitter on 429 / 503 / network errors
  * automatic re-registration when your token stops being accepted (401)
  * typed exceptions for the outcomes you have to make decisions about

Copyright 2026 The CoGNETs Consortium
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, Optional

import httpx

LOG = logging.getLogger("agent.client")

# Retry policy for transient failures. The graded arena injects 503s and 429s,
# so these numbers are the difference between finishing a run and not.
#
# BACKOFF_CAP is deliberately well under one round (4s in the graded scenario).
# A backoff longer than a round guarantees you miss it, which turns a transient
# fault into a lost round - the opposite of resilience.
MAX_ATTEMPTS = 5
BACKOFF_BASE = 0.25
BACKOFF_CAP = 1.5

# The critical path - reading the round and placing the bid - runs against the
# round's own clock instead. An injected 503 says nothing about the next
# request (the faults are independent dice rolls), so waiting the suggested
# full second three times in a row is how a round is lost: measured, it was
# the only round our agent ever missed. Against a deadline we back off
# exponentially with full jitter from a much smaller base, never sleep past the
# close, and keep trying for as long as the round is open.
FAST_BACKOFF_BASE = 0.06
FAST_BACKOFF_CAP = 0.5
DEADLINE_GUARD = 0.15   # stop retrying this long before the round closes


class ArenaClientError(Exception):
    """Base class for everything this client raises."""


class NotAdmissible(ArenaClientError):
    """403 - your node may not bid this round (flat battery, or ejected).

    Not fatal. Sit the round out and try the next one.
    """


class AlreadyBid(ArenaClientError):
    """409 - you already submitted for this round. Harmless; skip ahead."""


class RoundClosed(ArenaClientError):
    """410 - the round shut before your bid landed. Do not retry it."""


class WrongRound(ArenaClientError):
    """422 - you bid on a round that is no longer open. Re-read /v1/round."""


class NoRound(ArenaClientError):
    """404 - no round is open yet, or the round you asked about has not settled."""


class Ejected(ArenaClientError):
    """Your node has been ejected from the run. This one IS fatal."""


class ArenaClient:
    """A thin, resilient wrapper around the arena's REST API."""

    def __init__(
        self,
        base_url: str,
        team: str,
        timeout: float = 10.0,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.team = team
        self.max_attempts = max_attempts
        self.token: Optional[str] = None
        self.node_id: Optional[str] = None
        self.profile: Dict[str, Any] = {}
        self.arena_info: Dict[str, Any] = {}
        self._http = httpx.Client(timeout=timeout)
        self._rng = random.Random()

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ArenaClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --------------------------------------------------------------- plumbing
    def _headers(self) -> Dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _sleep_backoff(self, attempt: int, retry_after: Optional[str]) -> None:
        """Exponential backoff with full jitter, capped below one round.

        We honour Retry-After but still clamp it: the arena suggests 1 second,
        and blindly obeying a larger value would cost you the round.
        """
        if retry_after:
            try:
                time.sleep(min(float(retry_after), BACKOFF_CAP))
                return
            except (TypeError, ValueError):
                pass
        delay = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_CAP)
        time.sleep(delay * (0.5 + self._rng.random()))  # full jitter

    def _sleep_fast(self, attempt: int, deadline: Optional[float]) -> bool:
        """Deadline-aware backoff for the critical path.

        Exponential with full jitter from a small base, and never past the
        deadline (a `time.monotonic()` value). Returns False when there is no
        time left to try again.
        """
        delay = min(FAST_BACKOFF_BASE * (2 ** attempt), FAST_BACKOFF_CAP)
        delay *= 0.5 + self._rng.random()
        if deadline is not None:
            left = deadline - DEADLINE_GUARD - time.monotonic()
            if left <= 0.0:
                return False
            delay = min(delay, left)
        time.sleep(delay)
        return True

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        auth: bool = True,
        allow_reregister: bool = True,
        fast: bool = False,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One API call with retries.

        `fast=True` selects the critical-path policy: small jittered backoff,
        and with a `deadline` it keeps retrying until that moment rather than
        for a fixed number of attempts.
        """
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        attempts = 60 if (fast and deadline is not None) else self.max_attempts

        for attempt in range(attempts):
            try:
                response = self._http.request(
                    method, url, json=json, headers=self._headers() if auth else {}
                )
            except httpx.HTTPError as exc:
                last_error = exc
                LOG.debug("network error on %s %s: %s", method, path, exc)
                if fast:
                    if not self._sleep_fast(attempt, deadline):
                        break
                else:
                    self._sleep_backoff(attempt, None)
                continue

            status = response.status_code

            if 200 <= status < 300:
                return response.json() if response.content else {}

            detail = _detail(response)

            if status in (429, 503):
                LOG.debug("transient %s on %s (attempt %d)", status, path, attempt + 1)
                if fast:
                    if not self._sleep_fast(attempt, deadline):
                        break
                else:
                    self._sleep_backoff(attempt, response.headers.get("retry-after"))
                continue

            if status == 401 and auth and allow_reregister:
                LOG.info("token rejected, re-registering")
                self.register()
                continue

            if status == 403:
                if "ejected" in detail.lower():
                    raise Ejected(detail)
                raise NotAdmissible(detail)
            if status == 409:
                raise AlreadyBid(detail)
            if status == 410:
                raise RoundClosed(detail)
            if status == 422 and path.startswith("/v1/bid"):
                raise WrongRound(detail)
            if status == 404:
                raise NoRound(detail)

            raise ArenaClientError(f"{method} {path} -> {status}: {detail}")

        raise ArenaClientError(
            f"{method} {path} failed after retrying"
            + (f": {last_error}" if last_error else "")
        )

    # -------------------------------------------------------------- registry
    def register(self) -> Dict[str, Any]:
        """Join the swarm (or rejoin after an expiry) and store the token."""
        data = self._request(
            "POST",
            "/v1/register",
            json={"team": self.team},
            auth=False,
            allow_reregister=False,
        )
        self.token = data["token"]
        self.node_id = data["node_id"]
        self.profile = data.get("profile", {})
        self.arena_info = data.get("arena", {})
        LOG.info(
            "registered as %s (%s)  weights=%s  q_min=%.3f  s_min=%.3f",
            self.team,
            self.node_id,
            self.profile.get("weights"),
            self.profile.get("q_min", 0.0),
            self.profile.get("s_min", 0.0),
        )
        return data

    def heartbeat(self) -> Dict[str, Any]:
        return self._request("POST", "/v1/heartbeat")

    # --------------------------------------------------------------- auction
    def get_round(self) -> Dict[str, Any]:
        """Read the open round. Fast retries: seeing a new round late eats its window.

        Stamps the payload with `_deadline`, the round's close on our own
        monotonic clock (from `seconds_remaining`, so host clock skew between
        us and the arena does not matter).
        """
        payload = self._request("GET", "/v1/round", fast=True)
        payload["_deadline"] = time.monotonic() + float(payload.get("seconds_remaining", 0.0))
        self._refresh_live_features(payload)
        return payload

    def _refresh_live_features(self, payload: Dict[str, Any]) -> None:
        """Keep the volatile parts of ``profile["features"]`` current.

        Most of your profile is hardware and never changes. Two entries are not
        hardware: your battery, and your compromise score. The arena reports
        both in every round, and this copies them into `profile["features"]`
        so that `profile` tells you the truth at the moment you are asked to
        bid.

        This matters more than it looks. Your battery is the single most
        important number in the challenge - the energy you win comes out of it,
        and a flat battery costs you whole rounds - and `profile["features"]`
        is the first place anyone looks for it. Left frozen at its registration
        value, an agent that throttles its energy as charge falls would simply
        never fire, and would score the same as one that had never had the
        idea.

        `history[-1]["battery"]` is the other place to read it, and it is the
        one to use if you want the whole trace rather than the latest value.
        """
        you = payload.get("you")
        if not isinstance(you, dict):
            return
        features = self.profile.setdefault("features", {})
        for key in ("battery", "compromise"):
            if key in you:
                features[key] = you[key]

    def post_bid(self, round_index: int, bid: Dict[str, float],
                 deadline: Optional[float] = None) -> Dict[str, Any]:
        """Submit a bid, retrying until the round's `deadline` if one is given."""
        return self._request(
            "POST", "/v1/bid", json={"round": round_index, "bid": bid},
            fast=True, deadline=deadline,
        )

    def get_result(self, round_index: int, attempts: int = 2) -> Dict[str, Any]:
        """Fetch a settled round's result.

        Deliberately fewer retries than a bid: this is bookkeeping, not the
        critical path, and it must never eat the window for the next bid.
        """
        saved, self.max_attempts = self.max_attempts, attempts
        try:
            # Fast policy: two slow retries here once cost a whole round.
            return self._request("GET", f"/v1/result/{round_index}", fast=True)
        finally:
            self.max_attempts = saved

    # ----------------------------------------------------------------- views
    def me(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/me")

    def leaderboard(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/leaderboard", auth=False)

    def status(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/status", auth=False)

    def swarm(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/swarm", auth=False)

    def wait_for_arena(self, timeout: float = 60.0) -> bool:
        """Block until /healthz answers, so compose start order does not matter."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = self._http.get(f"{self.base_url}/healthz")
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        return False


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict):
        return str(payload.get("detail", payload))
    return str(payload)
