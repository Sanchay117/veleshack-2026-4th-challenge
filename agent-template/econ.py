"""
The arena's game mathematics, restated for the agent.

These mirror `arena/game.py` exactly (Kelly allocation, CES utility with
rho = 0.5, the two service floors) so the agent can evaluate a candidate bid
offline, before it is sent. The arena source ships with the challenge and is
the specification; this module is a faithful, vectorised copy of the parts an
agent needs to reason about, nothing more.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

RESOURCES = ("compute", "energy", "security")
EPS = 1e-9


def share(bid, others, capacity):
    """Kelly share of one pool: ``x = C * b / (b + S)``.

    Works on scalars and numpy arrays alike. When nobody else is bidding
    (``S == 0``) any positive bid takes the whole pool, which is exactly what
    the arena does.
    """
    bid = np.asarray(bid, dtype=float)
    total = bid + others
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(total > EPS, capacity * bid / np.maximum(total, EPS), 0.0)
    return np.where(bid > 0.0, x, 0.0)


def bid_for_share(target, others, capacity):
    """Invert the Kelly rule: the smallest bid that buys share ``target``.

    ``b = S * t / (C - t)``. Infinite when the target is the whole pool or
    more and somebody else is bidding.
    """
    if others <= EPS:
        return 1e-6 if target > 0 else 0.0
    if target >= capacity:
        return float("inf")
    return others * target / (capacity - target)


def ces(xc, xe, xs, w):
    """CES utility with rho = 0.5: ``(sum w_k * sqrt(x_k))^2``."""
    acc = (
        w["compute"] * np.sqrt(np.maximum(xc, 0.0))
        + w["energy"] * np.sqrt(np.maximum(xe, 0.0))
        + w["security"] * np.sqrt(np.maximum(xs, 0.0))
    )
    return acc * acc


def floor_factor(xc, xs, q_min, s_min):
    """Halve per missed floor, exactly as `arena.game.apply_floors` does."""
    f = np.where(xc + EPS < q_min, 0.5, 1.0)
    return f * np.where(xs + EPS < s_min, 0.5, 1.0)


def realised_utility(
    bid: Mapping[str, float],
    others: Mapping[str, float],
    capacities: Mapping[str, float],
    weights: Mapping[str, float],
    q_min: float,
    s_min: float,
) -> Dict[str, float]:
    """Score one concrete bid against a known field. Used for logging and tests."""
    x = {k: float(share(bid.get(k, 0.0), others.get(k, 0.0), capacities[k]))
         for k in RESOURCES}
    raw = float(ces(x["compute"], x["energy"], x["security"], weights))
    factor = float(floor_factor(x["compute"], x["security"], q_min, s_min))
    return {"utility": raw * factor, "utility_raw": raw, **{f"x_{k}": v for k, v in x.items()}}
