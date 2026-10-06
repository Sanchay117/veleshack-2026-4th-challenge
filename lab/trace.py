"""
Print one run round by round: who was awake, what we spent, what we scored.

    python lab/trace.py --seed 2 --team team-0 --strategy ours

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse

import sim


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--team", default="team-0")
    p.add_argument("--strategy", default="ours")
    p.add_argument("--scenario", default="graded")
    a = p.parse_args()

    r = sim.run(a.seed, a.team, a.strategy, a.scenario, keep_rounds=True)
    me = a.team
    print(f"{'rnd':>3} {'bat':>5} {'awake':<8} {'efrac':>6} {'x_E':>6} {'util':>6} "
          f"{'bots (battery)':<32} {'err':>7} {'ms':>5}")
    for row in r.rounds:
        n = row["nodes"]
        mine = n[me]
        b = row["brain"] or {}
        bots = "  ".join(f"{t[4:8]}:{v['battery']:.2f}{'' if v['awake'] else 'z'}"
                         for t, v in n.items() if t != me)
        print(f"{row['round']:>3} {mine['battery']:>5.2f} {'yes' if mine['awake'] else 'REST':<8} "
              f"{b.get('energy_fraction', 0):>6.3f} {b.get('x_energy', 0):>6.3f} "
              f"{mine['utility']:>6.3f} {bots:<32} {b.get('model_error', 0):>7.4f} "
              f"{b.get('think_ms', 0):>5.0f}")
    print(f"\nscore {r.score:.2f}   bots " +
          "  ".join(f"{k}={v:.2f}" for k, v in r.bots.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
