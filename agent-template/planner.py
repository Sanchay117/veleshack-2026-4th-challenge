"""
The planner: how much battery to spend this round, and how to split the budget.

Two layers.

1. **One round, exactly.** For a known field S_k, capacities C_k and budget W,
   `round_menu` lists every way we could spend: for each candidate energy bid,
   the best split of the remaining budget between compute and security, with
   both service floors priced in. The result is a menu of
   (utility this round, battery it costs) pairs.

2. **The whole run, by dynamic programming.** Battery is a stock we draw down
   and refill only by resting, and a round spent resting scores nothing. So
   the right spend this round depends on every round still to come. `plan`
   solves that as a finite-horizon DP over battery charge,

       V_t(B) = V_{t+1}(B + r)                                   if B <= cutoff
       V_t(B) = max_a  menu_t[a].utility + V_{t+1}(B - menu_t[a].drain)  otherwise

   where the future menus come from a forecast of who will be awake (see
   `strategy.forecast`). The DP picks when to sip energy, when to spend it,
   when a deliberate rest pays for itself, and it empties the battery on the
   last round instead of leaving charge on the table.

Everything is vectorised with numpy; a full 60-round plan takes a few
milliseconds, comfortably inside the 4 s bidding window.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from econ import EPS, bid_for_share, ces, floor_factor, share

#: Energy bid as a fraction of the budget. Dense near zero, because the first
#: sliver of energy is where sqrt(x) pays the most.
ENERGY_FRACTIONS = np.array([
    0.0, 0.002, 0.004, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.065,
    0.08, 0.1, 0.12, 0.15, 0.18, 0.22, 0.26, 0.3, 0.35, 0.42, 0.5, 0.6, 0.75,
])

#: Compute share of what is left after energy. The two floor-binding splits
#: are appended per row, so floors are bought exactly rather than to the grid.
SPLITS = np.linspace(0.0, 1.0, 161)

#: Battery grid for the DP.
BATTERY_GRID = np.linspace(0.0, 1.0, 401)


@dataclass
class Market1:
    """Everything that decides one round's menu."""

    budget: float
    capacities: Dict[str, float]
    field: Dict[str, float]          # S_k: everyone else's total bid


@dataclass
class Device:
    weights: Dict[str, float]
    q_min: float
    s_min: float
    mobility: float


@dataclass
class Menu:
    utility: np.ndarray      # (A,)
    drain: np.ndarray        # (A,)
    bids: np.ndarray         # (A, 3) compute, energy, security
    x_energy: np.ndarray     # (A,)


def round_menu(m: Market1, dev: Device, drain_rate: float, idle_drain: float,
               floor_margin: float = 0.0) -> Menu:
    """Best achievable utility for each candidate energy spend, in one round."""
    W = float(m.budget)
    C = m.capacities
    S = m.field

    b_e = ENERGY_FRACTIONS * W                                   # (A,)
    x_e = share(b_e, S["energy"], C["energy"])                   # (A,)
    rest = np.maximum(W - b_e, 0.0)                              # (A,)

    # Splits, plus the exact floor-buying points for each row.
    q_t = dev.q_min * (1.0 + floor_margin)
    s_t = dev.s_min * (1.0 + floor_margin)
    need_c = bid_for_share(q_t, S["compute"], C["compute"])
    need_s = bid_for_share(s_t, S["security"], C["security"])
    with np.errstate(divide="ignore", invalid="ignore"):
        s_floor_c = np.where(rest > EPS, need_c / np.maximum(rest, EPS), 2.0)
        s_floor_s = np.where(rest > EPS, 1.0 - need_s / np.maximum(rest, EPS), -1.0)
    extra = np.stack([s_floor_c, s_floor_s], axis=1)
    splits = np.concatenate([np.broadcast_to(SPLITS, (len(rest), len(SPLITS))), extra], axis=1)
    splits = np.clip(splits, 0.0, 1.0)                            # (A, K)

    b_c = splits * rest[:, None]
    b_s = (1.0 - splits) * rest[:, None]
    x_c = share(b_c, S["compute"], C["compute"])
    x_s = share(b_s, S["security"], C["security"])
    u = ces(x_c, x_e[:, None], x_s, dev.weights)
    u = u * floor_factor(x_c, x_s, q_t, s_t)                      # (A, K)

    best = np.argmax(u, axis=1)
    rows = np.arange(len(rest))
    bids = np.stack([b_c[rows, best], b_e, b_s[rows, best]], axis=1)
    drain = drain_rate * x_e * (1.0 + dev.mobility) + idle_drain
    return Menu(utility=u[rows, best], drain=drain, bids=bids, x_energy=x_e)


@dataclass
class Plan:
    action: int                    # index into ENERGY_FRACTIONS for this round
    menu: Menu                     # this round's menu
    q_values: np.ndarray           # value of each action, now
    battery_path: List[float]      # projected start-of-round battery, this round on
    action_path: List[int]         # projected action per round (-1 = resting)
    shadow_price: float            # dV/dB: what one unit of charge is worth now
    value: float                   # expected utility still to come


def plan(battery: float, menus: Sequence[Menu], cutoff: float,
         recharge: float) -> Plan:
    """Solve the battery DP over the remaining rounds; menus[0] is this round."""
    grid = BATTERY_GRID
    horizon = len(menus)
    V_next = np.zeros_like(grid)
    policies: List[Optional[np.ndarray]] = [None] * horizon
    values: List[np.ndarray] = [V_next] * (horizon + 1)

    for t in range(horizon - 1, -1, -1):
        mu = menus[t]
        after = np.maximum(0.0, grid[:, None] - mu.drain[None, :])          # (B, A)
        q = mu.utility[None, :] + np.interp(after, grid, V_next)            # (B, A)
        act = np.argmax(q, axis=1)
        V = q[np.arange(len(grid)), act]
        resting = grid <= cutoff + 1e-12
        V = np.where(resting, np.interp(np.minimum(1.0, grid + recharge), grid, V_next), V)
        act = np.where(resting, -1, act)
        policies[t] = act
        values[t] = V
        V_next = V

    # This round, evaluated at the exact charge rather than on the grid.
    mu0 = menus[0]
    after0 = np.maximum(0.0, battery - mu0.drain)
    V1 = values[1] if horizon > 1 else np.zeros_like(grid)
    q0 = mu0.utility + np.interp(after0, grid, V1)
    a0 = int(np.argmax(q0))

    # Shadow price of charge: the slope of the value function where we stand.
    lo, hi = max(0.0, battery - 0.02), min(1.0, battery + 0.02)
    shadow = float((np.interp(hi, grid, values[0]) - np.interp(lo, grid, values[0]))
                   / max(hi - lo, 1e-6))

    # Roll the policy forward to show the plan.
    path, acts = [], []
    b = battery
    for t in range(horizon):
        path.append(round(float(b), 4))
        if t == 0:
            a = a0
        else:
            a = int(policies[t][int(np.argmin(np.abs(grid - b)))])
        acts.append(a)
        if a < 0 or b <= cutoff + 1e-12:
            b = min(1.0, b + recharge)
        else:
            b = max(0.0, b - float(menus[t].drain[a]))

    return Plan(action=a0, menu=mu0, q_values=q0, battery_path=path,
                action_path=acts, shadow_price=shadow, value=float(q0[a0]))
