# CoGNETs Swarm Arena

The environment for the VelesHack 2026 challenge **“Dynamic Node Registration
and Auction-Based Resource Allocation in an Edge Computing Environment”**.

You write one container: an agent that joins a decentralised edge swarm,
survives in it, and competes each round for shares of three scarce resource
pools under a budget. Everything else here is provided, runs on your laptop,
and is open source so you can read exactly how you are being scored.

```bash
git clone https://github.com/czavitsanos-iti/veleshack-2026-4th-challenge
cd veleshack-2026-4th-challenge
cp .env.example .env
sed -i 's/^TEAM_NAME=.*/TEAM_NAME=team-example/' .env   # use your own team name
grep TEAM_NAME .env                                     # check it took
make up                                                 # arena + three bots -> localhost:8080
make agent                                              # you, now on the leaderboard and losing
```

Then edit **one file** — `agent-template/strategy.py` — and beat the bots.

---

## Read these, in this order

| | |
|---|---|
| **[docs/quickstart.md](quickstart.md)** | get running, and the fixes for the five things that go wrong |
| **[docs/strategy-primer.md](strategy-primer.md)** | **the important one.** Where the wins actually are |
| **[docs/api.md](api.md)** | every endpoint, every status code |
| **[docs/cognets-mapping.md](cognets-mapping.md)** | what this is a reduction of, and what was left out |

Interactive API docs are served at `localhost:8080/docs` once the arena is up.

---

## The game in one screen

Each round the arena publishes the pool sizes `C_k`, the prices that cleared
last round `λ_k`, and your budget `W`. You submit a bid:

```
{ "compute": b_C, "energy": b_E, "security": b_S }      with   Σ b ≤ W
```

Pools are shared out in proportion to bids, and your payoff is a CES utility
over what you won:

```
x_k = C_k · b_k / Σ_j b_j          u = ( Σ_k w_k · x_k^0.5 ) ^ 2
```

Three things make that harder than it looks:

- **Two service floors.** Miss your minimum compute or security share and the
  round's utility is halved. Miss both and it is quartered.
- **Energy is not free.** The energy share you *win* drains your own battery.
  Flatten it and your node sits whole rounds out, scoring nothing.
- **The swarm does not sit still.** Leases expire, capacity moves, and the
  graded arena returns `503`s and adds latency.

Your score is the sum of your per-round utility. The bar is to finish ahead of
all three baseline bots.

---

## What you are up against

| Bot | What it does | Why it is here |
|---|---|---|
| `naive-max` | spends everything, split by its weights | the floor. Also what the template ships with |
| `even-split` | one third on each pool | a control, and stubborner than it looks |
| `proportional` | leans toward big/cheap pools and buys its service floors | the most sophisticated of the three, and barely ahead |

None of them knows that the energy it wins drains its own battery. That
omission is deliberate — it is the largest single source of score in the
challenge, and a bot that already knew it would mean you scored nothing for
working it out.

How much it costs them is worth knowing before you start. Over 30 runs, all
three bots spend **roughly a third of the run** flat on their backs, scoring
nothing; the organisers' reference agent spends three rounds. Reading the
market carefully, as `proportional` does, is worth about 1.5% over spending
blindly. The battery is worth an order of magnitude more than that.

All three are in `baselines/bot.py`. Read them.

---

## Commands

| | |
|---|---|
| `make up` | arena + the three bots |
| `make agent` | build and run your agent |
| `make graded` | restart on the hostile scenario you are scored on |
| `make check` | **the conformance suite the organisers run.** Run it before submitting |
| `make check-docker IMAGE=...` | the same suite against your built image |
| `make board` / `make status` | the standings / the run state |
| `make down` / `make reset` | stop / wipe and restart |

`make help` lists everything.

No Docker on your laptop? `pip install -e . && python -m arena` is a fully
supported path — see the quickstart.

---

## Layout

```
arena/              the environment: registry, auctioneer, scorer, leaderboard
  game.py             Kelly allocation, CES utility, floors, prices, LSW
  state.py            nodes, leases, rounds, battery, scoring
  scenarios/          practice.yaml and graded.yaml
agent-template/     your agent
  strategy.py         <- the only file you must change
  agent.py            the main loop. Works already. Read it, don't rewrite it
  client.py           HTTP client with retry/backoff. Works already
baselines/          the three bots
tests/              the conformance suite the organisers grade with
docs/               quickstart, strategy primer, API, CoGNETs mapping
```

---

## Evaluation

Graded after the hacking window, in a fresh arena alongside the same three
bots, over **four runs on four seeds that are not published in advance**. One
run is too short to separate a good idea from a good draw, so every number
below is pooled across all four.

| Criterion | Pts | What we check |
|---|--:|---|
| Functional core | 30 | Builds from a clean clone; registers, heartbeats and bids validly in ≥95% of the rounds you are admissible for; completes the run |
| Resilience | 20 | Survives injected 503/429, latency, lease expiry and battery outage. Retries with backoff |
| Strategy | 25 | Cumulative score, scaled from the template's strategy up to our reference agent |
| Engineering | 15 | Env-var config, no hardcoded URLs or secrets, readable code, useful logs, accurate README |
| Insight & pitch | 10 | What you tried, what you measured, what you learned. Honest negative results count, and so does measuring your effect on the rest of the swarm |

**How the strategy points work.** Let `S` be your cumulative score, `T` the
score of the strategy the template ships with, and `R` the score of the
organisers' reference agent:

```
strategy_points = 25 · clip( (S − T) / (R − T), 0, 1 )
```

Change nothing and you score 0. Match the reference and you get all 25.
Everything in between earns partial credit, so an improvement that does not
quite beat every bot is still worth points. You are not competing against the
other teams for these — every team that reaches the reference gets all of them.

`T` and `R` are not other agents standing in the arena next to you. They are
**counterfactuals**: after your run, we replay the same seed, the same three
bots and *your own device* twice more, once with the template's strategy in
your place and once with the reference agent. So the two numbers your score is
measured against answer one question — what would those strategies have scored
on your machine, in your market? — and nothing about them depends on your
device draw, on the fault dice, or on how you happened to bid. They can be
recomputed from the seed alone, which is how we answer a query about a score.

Two consequences worth knowing:

- A run where the reference fails to beat the template is a bad *seed*, not a
  bad team. Those runs are dropped, and the remaining ones carry the score.
- Rounds you lose to injected faults cost you resilience points, not strategy
  points, and the bots are never faulted — so the market is the same in all
  three runs.

Rounds you spend resting on a flat battery are **not** counted against your
functional score. They cost you score instead, which is the point.

Ties are broken by the strategy score, then by submission time. The most
interesting analysis will not always come from the top of the leaderboard, which
is what the ten **Insight & pitch** points are there for.

**A question worth answering in your write-up.** The arena reports the swarm's
log social welfare every round, `Σ_i log(u_i)`, and shows every node's score on
the leaderboard. Did your strategy make the swarm better off, or did it take
from its neighbours? Measure it and say so. That analysis counts under Insight.

---

## Submission

On [taikai.network](https://taikai.network/), before the deadline announced at
kick-off:

- [ ] A link to a public Git repository — your fork of this one, with your work
      committed
- [ ] Your `TEAM_NAME`, exactly as you used it all weekend. It decides your device
      profile, so the graded run has to use the same one
- [ ] The path to your Dockerfile, normally `agent-template/Dockerfile`, and the
      build command if it is not a plain `docker build .`
- [ ] Your README, with the 300-word strategy write-up and all team members named.
      What you tried, what you measured, what did not work — `docs/strategy-primer.md`
      section 8 says what earns the Insight marks
- [ ] A three-minute demo video, or confirmation that you will pitch live
- [ ] Optionally, a screenshot or `results.json` from your best local run

Before you submit, check the boring things — this is where strong teams lose
points. Your repository must clone and build on a machine that has never seen
your laptop: no `.env` committed, no absolute paths, every dependency you
installed by hand actually in `requirements.txt`. Run `make check` first.

---

## Rules worth knowing before you start

- **Reading the arena source is encouraged, not cheating.** It ships with the
  challenge on purpose. What will not work is hardcoding outcomes: the graded
  run uses a seed that is not published in advance.
- **Any language is fine** if the deliverable is a Docker image. Only Python is
  supported by the template and the mentor.
- One agent per team, one registration per agent. Farming resources under
  several names is detected and disqualifies the run.
- Malformed, negative or over-budget bids raise your compromise score κ. Twelve
  of them and you are ejected for the rest of the run. Validate before you send.

---

## Licence

Apache-2.0. See [LICENSE](../LICENSE).

Copyright 2026 The CoGNETs Consortium.
