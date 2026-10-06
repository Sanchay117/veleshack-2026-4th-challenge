# The agent (team NullPointerException)

This folder is the deliverable: the Docker image is built from it, with a
plain `docker build .` and no arguments.

| File | Role |
|---|---|
| `agent.py` | Main loop (template's, extended): registration, heartbeat thread, one swarm snapshot per round, bid before bookkeeping, decision logs, clean shutdown |
| `client.py` | HTTP client (template's, extended): deadline-aware jittered retries on the critical path, auto re-registration, typed exceptions |
| `strategy.py` | The decision: forecast the swarm, plan the battery, buy the floors exactly. `decide_bid` keeps the template signature |
| `opponents.py` | Opponent model: shadows of the three baseline bots, residual for anything unexplained, battery physics fitted online |
| `planner.py` | Per-round menu of (utility, battery cost) and the battery dynamic programme |
| `econ.py` | The arena's game maths, restated for the agent |
| `telemetry.py`, `dashboard/` | Optional live dashboard, off unless `TELEMETRY_PORT` is set |
| `runner.py` | Template's compact loop, used by the baseline bots |

## Run it

```bash
# from the repository root, with the arena already up (make up)
make agent          # in Docker
make demo           # in Docker, with the dashboard on http://localhost:8090

# or natively
pip install -r requirements.txt
ARENA_URL=http://localhost:8080 TEAM_NAME=NullPointerException python agent.py
```

| Variable | Default | |
|---|---|---|
| `ARENA_URL` | `http://localhost:8080` | arena base URL |
| `TEAM_NAME` | `unnamed-team` | decides the device profile; ours is `NullPointerException` |
| `LOG_LEVEL` | `INFO` | `DEBUG` adds every HTTP request |
| `TELEMETRY_PORT` | unset (off) | serve the dashboard and `/api/state` on this port |

## A decision, as logged

```
round 23  battery=0.142  awake=2  bid=C0.512/E0.061/S0.391  x_E=0.086  exp_u=0.403  shadow=0.84  model=shadow(err 0.0000)  41ms
```

Battery at the start of the round, how many rivals are awake, the bid, the
energy share it buys, the utility the planner expects, the value of one unit
of charge right now, the opponent model's measured error, and think time.

## Before submitting

```bash
make check                                              # conformance, as a process
docker build -t nullpointerexception/agent .            # from this folder
make check-docker IMAGE=nullpointerexception/agent      # conformance, the image itself
make test                                               # unit tests
```
