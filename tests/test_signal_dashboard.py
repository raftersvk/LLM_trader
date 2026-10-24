"""Tests for the standalone, read-only signal dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.signal_dashboard.app import create_app


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _signal(
    decision_id: str,
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    created_at: datetime,
    action: str = "BUY",
    confidence: float = 0.9,
    jev: dict | None = None,
) -> dict:
    candle_open = created_at.replace(minute=0, second=0, microsecond=0)
    timeframe_delta = timedelta(hours=1)
    return {
        "schema_version": "1.0",
        "decision_id": decision_id,
        "source": "llm_trader",
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_open_time": _iso(candle_open),
        "candle_close_time": _iso(candle_open + timeframe_delta),
        "created_at": _iso(created_at),
        "expires_at": _iso(candle_open + 2 * timeframe_delta),
        "action": action,
        "confidence": confidence,
        "risk_approved": True,
        "sl": 60000.0,
        "tp": 70000.0,
        "position_size_pct": 0.1,
        "model": "test-model",
        "provider": "test-provider",
        "latency_ms": 123,
        "reasoning": "test reasoning",
        "jev": jev,
    }


def _write_jsonl(directory: Path, name: str, rows: list) -> None:
    lines = [json.dumps(row) if isinstance(row, dict) else str(row) for row in rows]
    (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(directory: Path, name: str, payload: dict) -> None:
    (directory / name).write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _client(directory: Path, **kwargs) -> TestClient:
    return TestClient(create_app(signals_dir=str(directory), **kwargs))


def test_root_page_and_no_mutating_routes(tmp_path):
    _write_jsonl(tmp_path, "BTC_USDT_1h.jsonl", [])

    with _client(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Signal Chain" in page.text
        assert "READ-ONLY" in page.text

        assert client.post("/api/signals").status_code == 405
        assert client.delete("/api/health").status_code == 405
        assert client.put("/api/pairs").status_code == 405


def test_signals_are_newest_first_and_respect_limit(tmp_path):
    base = datetime(2030, 1, 1, tzinfo=timezone.utc)
    _write_jsonl(
        tmp_path,
        "BTC_USDT_1h.jsonl",
        [
            _signal("older", created_at=base),
            _signal("newest", created_at=base + timedelta(hours=2)),
            _signal("middle", created_at=base + timedelta(hours=1)),
        ],
    )

    with _client(tmp_path) as client:
        response = client.get("/api/signals", params={"limit": 2})
        assert response.status_code == 200
        body = response.json()
        assert [row["decision_id"] for row in body] == ["newest", "middle"]
        assert body[0]["symbol"] == "BTC/USDT"
        assert body[0]["timeframe"] == "1h"
        assert body[0]["age_seconds"] is not None
        assert body[0]["expired"] is False

        all_rows = client.get("/api/signals").json()
        assert [row["decision_id"] for row in all_rows] == [
            "newest",
            "middle",
            "older",
        ]


def test_default_limit_comes_from_app_configuration(tmp_path):
    base = datetime(2030, 1, 1, tzinfo=timezone.utc)
    _write_jsonl(
        tmp_path,
        "BTC_USDT_1h.jsonl",
        [
            _signal(f"signal-{index}", created_at=base + timedelta(hours=index))
            for index in range(5)
        ],
    )

    with _client(tmp_path, limit=2) as client:
        assert len(client.get("/api/signals").json()) == 2


def test_signals_filtered_by_symbol_and_timeframe(tmp_path):
    base = datetime(2030, 1, 1, tzinfo=timezone.utc)
    _write_jsonl(
        tmp_path,
        "BTC_USDT_1h.jsonl",
        [_signal("btc", created_at=base, symbol="BTC/USDT", timeframe="1h")],
    )
    _write_jsonl(
        tmp_path,
        "ETH_USDT_4h.jsonl",
        [
            _signal(
                "eth",
                created_at=base,
                symbol="ETH/USDT",
                timeframe="4h",
                action="SELL",
                confidence=0.65,
            )
        ],
    )

    with _client(tmp_path) as client:
        only_btc = client.get("/api/signals", params={"symbol": "BTC/USDT"}).json()
        assert [row["decision_id"] for row in only_btc] == ["btc"]

        only_4h = client.get("/api/signals", params={"timeframe": "4h"}).json()
        assert [row["decision_id"] for row in only_4h] == ["eth"]


def test_malformed_and_partial_lines_are_skipped_and_counted(tmp_path):
    base = datetime(2030, 1, 1, tzinfo=timezone.utc)
    _write_jsonl(
        tmp_path,
        "BTC_USDT_1h.jsonl",
        [
            _signal("good-1", created_at=base),
            "{not valid json",
            '{"decision_id": "partial"}',
            "",
            _signal("good-2", created_at=base + timedelta(hours=1)),
            "[1, 2, 3]",
        ],
    )

    with _client(tmp_path) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        assert health["counts"]["signals"] == 2
        assert health["counts"]["skipped_lines"] == 3

        rows = client.get("/api/signals").json()
        assert [row["decision_id"] for row in rows] == ["good-2", "good-1"]


def test_missing_signals_directory_never_crashes(tmp_path):
    missing = tmp_path / "does-not-exist"

    with _client(missing) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["status"] == "missing"
        assert payload["counts"]["signals"] == 0
        assert payload["counts"]["skipped_lines"] == 0

        assert client.get("/api/signals").json() == []
        assert client.get("/api/heartbeats").json() == []
        assert client.get("/api/pairs").json() == []


def test_heartbeat_fresh_vs_stale_and_default_two_times_timeframe(tmp_path):
    now = datetime.now(timezone.utc)
    _write_json(
        tmp_path,
        "heartbeat_BTC_USDT_1h.json",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "candle_open_time": _iso(now - timedelta(minutes=5)),
            "heartbeat_at": _iso(now - timedelta(seconds=10)),
        },
    )
    _write_json(
        tmp_path,
        "heartbeat_ETH_USDT_1h.json",
        {
            "symbol": "ETH/USDT",
            "timeframe": "1h",
            "candle_open_time": _iso(now - timedelta(hours=3)),
            "heartbeat_at": _iso(now - timedelta(hours=3)),
        },
    )

    with _client(tmp_path) as client:
        rows = client.get("/api/heartbeats").json()
        by_symbol = {row["symbol"]: row for row in rows}

        assert by_symbol["BTC/USDT"]["stale"] is False
        assert by_symbol["BTC/USDT"]["stale_after_seconds"] == 7200
        assert by_symbol["BTC/USDT"]["legacy"] is False
        assert by_symbol["ETH/USDT"]["stale"] is True
        assert by_symbol["ETH/USDT"]["age_seconds"] >= 3 * 3600 - 5

    # An explicit --stale-after overrides the timeframe-derived window.
    with _client(tmp_path, stale_after=5) as client:
        rows = client.get("/api/heartbeats").json()
        assert all(row["stale"] is True for row in rows)


def test_heartbeat_summary_is_not_reported_as_a_heartbeat(tmp_path):
    now = datetime.now(timezone.utc)
    _write_json(
        tmp_path,
        "heartbeat_BTC_USDT_1h.json",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "candle_open_time": _iso(now - timedelta(minutes=5)),
            "heartbeat_at": _iso(now - timedelta(seconds=10)),
        },
    )
    # Written every cycle by SignalService.write_heartbeat_summary; it has no
    # symbol/timeframe/heartbeat_at and must never surface as a phantom entry.
    _write_json(
        tmp_path,
        "heartbeat_summary.json",
        {
            "updated_at": _iso(now),
            "pairs": {"BTC/USDT:1h": _iso(now - timedelta(seconds=10))},
        },
    )

    with _client(tmp_path) as client:
        rows = client.get("/api/heartbeats").json()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "BTC/USDT"
        assert rows[0]["file"] == "heartbeat_BTC_USDT_1h.json"

        health = client.get("/api/health").json()
        assert health["counts"]["heartbeat_files"] == 1
        assert health["counts"]["heartbeats"] == 1

    summary_only = tmp_path / "summary-only"
    summary_only.mkdir()
    _write_json(summary_only, "heartbeat_summary.json", {"updated_at": _iso(now), "pairs": {}})

    with _client(summary_only) as client:
        assert client.get("/api/heartbeats").json() == []
        health = client.get("/api/health").json()
        assert health["counts"]["heartbeat_files"] == 0
        assert health["counts"]["heartbeats"] == 0


def test_legacy_shared_heartbeat_is_labelled_legacy(tmp_path):
    now = datetime.now(timezone.utc)
    _write_json(
        tmp_path,
        "heartbeat.json",
        {
            "symbol": "SOL/USDT",
            "timeframe": "15m",
            "candle_open_time": _iso(now - timedelta(minutes=15)),
            "heartbeat_at": _iso(now),
        },
    )

    with _client(tmp_path) as client:
        rows = client.get("/api/heartbeats").json()
        assert len(rows) == 1
        assert rows[0]["legacy"] is True
        assert rows[0]["symbol"] == "SOL/USDT"
        assert rows[0]["timeframe"] == "15m"
        assert rows[0]["stale"] is False


def test_heartbeat_filename_timeframe_parsing_for_slash_symbol(tmp_path):
    now = datetime.now(timezone.utc)
    # Payload intentionally omits symbol/timeframe: they must come from the
    # filename token heartbeat_BTC_USDT_1h.json.
    _write_json(
        tmp_path,
        "heartbeat_BTC_USDT_1h.json",
        {"candle_open_time": _iso(now), "heartbeat_at": _iso(now)},
    )

    with _client(tmp_path) as client:
        rows = client.get("/api/heartbeats").json()
        assert rows[0]["symbol"] == "BTC/USDT"
        assert rows[0]["timeframe"] == "1h"
        assert rows[0]["stale_after_seconds"] == 2 * 3600


def test_pairs_join_last_signal_and_heartbeat(tmp_path):
    now = datetime.now(timezone.utc)
    _write_jsonl(
        tmp_path,
        "BTC_USDT_1h.jsonl",
        [
            _signal(
                "latest",
                created_at=now - timedelta(minutes=10),
                action="SELL",
                confidence=0.35,
            ),
            _signal(
                "previous",
                created_at=now - timedelta(hours=1, minutes=10),
            ),
        ],
    )
    _write_json(
        tmp_path,
        "heartbeat_BTC_USDT_1h.json",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "candle_open_time": _iso(now - timedelta(minutes=10)),
            "heartbeat_at": _iso(now - timedelta(seconds=30)),
        },
    )

    with _client(tmp_path) as client:
        pairs = client.get("/api/pairs").json()
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["symbol"] == "BTC/USDT"
        assert pair["timeframe"] == "1h"
        assert pair["last_action"] == "SELL"
        assert pair["last_confidence"] == pytest.approx(0.35)
        assert pair["last_candle_open_time"] is not None
        assert pair["heartbeat_present"] is True
        assert pair["heartbeat_age_seconds"] is not None
        assert pair["stale"] is False


def test_invalid_timestamps_do_not_crash_responses(tmp_path):
    _write_jsonl(
        tmp_path,
        "BTC_USDT_1h.jsonl",
        [
            {
                "decision_id": "bad-time",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "candle_open_time": "not-a-timestamp",
                "created_at": "not-a-timestamp",
                "expires_at": "also-bad",
                "action": "HOLD",
                "confidence": 0.5,
                "reasoning": "unknown clock",
            }
        ],
    )
    _write_json(
        tmp_path,
        "heartbeat_BTC_USDT_1h.json",
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "heartbeat_at": "not-a-timestamp",
        },
    )

    with _client(tmp_path) as client:
        assert client.get("/api/health").status_code == 200
        signal = client.get("/api/signals").json()[0]
        assert signal["age_seconds"] is None
        assert signal["expired"] is False

        heartbeat = client.get("/api/heartbeats").json()[0]
        assert heartbeat["age_seconds"] is None
        assert heartbeat["stale"] is True


def test_jev_veto_is_exposed(tmp_path):
    now = datetime.now(timezone.utc)
    _write_jsonl(
        tmp_path,
        "BTC_USDT_1h.jsonl",
        [
            _signal(
                "vetoed",
                created_at=now,
                jev={"veto": True, "reason": "risk-off regime"},
            )
        ],
    )

    with _client(tmp_path) as client:
        row = client.get("/api/signals").json()[0]
        assert row["jev"]["reason"] == "risk-off regime"
        assert row["jev_veto"] is True
        assert row["jev_reason"] == "risk-off regime"
