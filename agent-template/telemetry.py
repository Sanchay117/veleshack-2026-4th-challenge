"""
Optional live telemetry: the agent's view of the run, for the mission-control
dashboard.

Off unless `TELEMETRY_PORT` is set, so the graded container behaves exactly
like a plain agent. When on, a small stdlib HTTP server (no extra
dependencies) serves:

    GET /             the dashboard (dashboard/index.html)
    GET /api/state    everything the dashboard draws, as JSON

The dashboard reads only from here. The arena sends no CORS headers, so the
agent relays the public leaderboard and status it already fetches each round.

Copyright 2026 Sanchay Singh
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("agent.telemetry")
DASHBOARD = Path(__file__).resolve().parent / "dashboard" / "index.html"
MAX_ROUNDS = 400


class Telemetry:
    """Thread-safe store of what the dashboard needs."""

    def __init__(self, team: str) -> None:
        self.team = team
        self._lock = threading.Lock()
        self._rounds: List[Dict[str, Any]] = []
        self._decision: Dict[str, Any] = {}
        self._decisions: Dict[int, Dict[str, Any]] = {}
        self._board: List[Dict[str, Any]] = []
        self._status: Dict[str, Any] = {}

    def reset(self) -> None:
        with self._lock:
            self._rounds.clear()
            self._decision = {}
            self._decisions = {}

    def on_board(self, round_index: int, board: Dict[str, Any]) -> None:
        """Called once per round with `/v1/leaderboard` (which embeds status)."""
        rows = board.get("leaderboard") or []
        status = board.get("status") or {}
        with self._lock:
            self._board, self._status = rows, status
            # The board seen as round r opens describes the state after r-1.
            settled = round_index - 1
            if settled >= 1 and not any(x["round"] == settled for x in self._rounds):
                lsw = (status.get("lsw_series") or [None])[-1]
                self._rounds.append({
                    "round": settled,
                    "lsw": lsw,
                    "nodes": {r["team"]: {"battery": r.get("battery"), "score": r.get("score"),
                                          "idle": r.get("rounds_idle"),
                                          "played": r.get("rounds_participated")}
                              for r in rows},
                })
                del self._rounds[:-MAX_ROUNDS]

    def on_decision(self, decision: Dict[str, Any]) -> None:
        with self._lock:
            self._decision = dict(decision)
            self._decisions[int(decision.get("round", 0))] = {
                k: decision.get(k) for k in
                ("energy_fraction", "x_energy", "expected_utility", "drain", "battery",
                 "shadow_price", "awake", "think_ms", "model", "model_error")
            }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "team": self.team,
                "decision": self._decision,
                "decisions": dict(self._decisions),
                "rounds": list(self._rounds),
                "leaderboard": self._board,
                "status": self._status,
            }


def serve(telemetry: Telemetry, port: int, host: str = "0.0.0.0") -> Optional[ThreadingHTTPServer]:
    """Start the server on a daemon thread. Never raises: telemetry is optional."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path.startswith("/api/state"):
                body = json.dumps(telemetry.snapshot(), default=float).encode()
                self._send(200, "application/json", body)
            elif self.path in ("/", "/index.html") and DASHBOARD.is_file():
                self._send(200, "text/html; charset=utf-8", DASHBOARD.read_bytes())
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            return

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        LOG.warning("telemetry disabled: cannot bind port %d (%s)", port, exc)
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    LOG.info("telemetry and dashboard on http://localhost:%d", port)
    return server
