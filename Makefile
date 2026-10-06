# CoGNETs Swarm Arena
#
#   make up        arena + the three baseline bots
#   make agent     build and run your agent
#   make check     the conformance suite the organisers also run
#   make graded    rehearse against the hostile scenario
#   make down      stop and reset everything
#
# Copyright 2026 The CoGNETs Consortium
# SPDX-License-Identifier: Apache-2.0

SHELL := /bin/bash
COMPOSE ?= docker compose
PY ?= python3

.DEFAULT_GOAL := help
.PHONY: help up down logs agent agent-logs check check-docker graded reset board \
        status install dev-arena dev-bots dev-agent lint clean test sim replay demo

## ---------------------------------------------------------------- docker ---

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Leaderboard: http://localhost:$${ARENA_PORT:-8080}"
	@echo "  API docs:    http://localhost:$${ARENA_PORT:-8080}/docs"

up:  ## Start the arena and the three baseline bots
	$(COMPOSE) up -d --build arena bot-naive-max bot-even-split bot-proportional
	@echo
	@echo "  Arena is up. Leaderboard: http://localhost:$${ARENA_PORT:-8080}"
	@echo "  Now run: make agent"

graded:  ## Restart the arena on the graded (hostile) scenario
	SCENARIO=graded $(COMPOSE) up -d --build --force-recreate \
	  arena bot-naive-max bot-even-split bot-proportional
	@echo "  Graded scenario running: chaos on, 60 rounds, shorter windows."

agent:  ## Build and run your agent (foreground, Ctrl-C to stop)
	$(COMPOSE) --profile agent up --build agent

agent-logs:  ## Follow your agent's logs
	$(COMPOSE) logs -f agent

logs:  ## Follow the arena's logs
	$(COMPOSE) logs -f arena

status:  ## Print the current run status
	@curl -fsS http://localhost:$${ARENA_PORT:-8080}/v1/status | $(PY) -m json.tool

board:  ## Print the leaderboard
	@curl -fsS http://localhost:$${ARENA_PORT:-8080}/v1/leaderboard \
	  | $(PY) -c "import json,sys; d=json.load(sys.stdin); \
[print(f\"{r['rank']:>2}  {r['team']:<24} {r['score']:>8.3f}  \
rounds={r['rounds_participated']:<3} missed={r['rounds_missed']:<3} \
floors={r['floor_violations']}\") for r in d['leaderboard']]"

down:  ## Stop everything and remove the containers
	$(COMPOSE) --profile agent down -v --remove-orphans

reset: down up  ## Wipe the run and start a fresh arena

## ------------------------------------------------------------------ check ---

check:  ## Run the conformance suite against your agent (what we grade on)
	$(PY) tests/conformance.py --agent-cmd "$(AGENT_CMD)"

check-docker:  ## The same suite against a built image: make check-docker IMAGE=you/agent
	@test -n "$(IMAGE)" || { echo "  set IMAGE, e.g. make check-docker IMAGE=team-kappa/agent:latest"; exit 2; }
	$(PY) tests/conformance.py --image "$(IMAGE)"

## ------------------------------------------- no-docker fallback (pip path) ---

install:  ## Install the arena locally, for laptops where Docker is blocked
	$(PY) -m pip install -e .

dev-arena:  ## Run the arena natively on :8080
	$(PY) -m arena --scenario $${SCENARIO:-practice}

dev-bots:  ## Run the three baseline bots natively (needs dev-arena running)
	@for b in naive-max even-split proportional; do \
	  BOT=$$b TEAM_NAME=bot-$$b ARENA_URL=http://localhost:8080 \
	    $(PY) baselines/bot.py & \
	done; wait

dev-agent:  ## Run your agent natively (needs dev-arena running)
	cd agent-template && ARENA_URL=http://localhost:8080 \
	  TEAM_NAME=$${TEAM_NAME:-unnamed-team} $(PY) agent.py

## ------------------------------------------------------- team NullPointer ---

test:  ## Unit tests for the agent (stdlib unittest, no extra dependencies)
	$(PY) -m unittest discover -s tests -p 'test_*.py'

sim:  ## Compare strategies in-process over many seeds (see lab/sim.py --help)
	$(PY) lab/sim.py --seeds 20 --teams 3 --strategy template taper ours ours-no-model ours-no-plan

replay:  ## Replay a simulated run in the dashboard on http://localhost:8090
	$(PY) lab/replay.py --seed $${SEED:-2} --team $${TEAM_NAME:-team-0} --speed $${SPEED:-0.8}

demo:  ## Live: arena + bots + our agent, dashboard on http://localhost:8090
	$(COMPOSE) up -d --build arena bot-naive-max bot-even-split bot-proportional
	$(COMPOSE) --profile agent up --build agent

## ------------------------------------------------------------------ chores ---

lint:  ## Byte-compile everything, to catch syntax errors early
	$(PY) -m compileall -q arena agent-template baselines tests

clean:  ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
