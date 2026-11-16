"""Undelivered background notifications are observable (one WARNING per event) and bounded
(given up after ``_MAX_DELIVERY_FAILURES`` failed redeliveries instead of requeueing forever)."""

import asyncio
import logging
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run_notifications as rn
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource

SESSION_KEY = "agent:main:telegram:dm:123:42"


class RefusingHandler(AsyncMock):
    """Fake transport that never issues the admission receipt (-> WakeNotAccepted)."""


class AdmittingHandler(AsyncMock):
    async def _execute_mock_call(self, event, *args, **kwargs):
        result = await super()._execute_mock_call(event, *args, **kwargs)
        event._gateway_accepted = True
        return result


def _build_runner(monkeypatch, tmp_path, handler) -> GatewayRunner:
    (tmp_path / "config.yaml").write_text("display:\n  background_process_notifications: all\n", encoding="utf-8")
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[Platform.TELEGRAM] = SimpleNamespace(send=AsyncMock(), handle_message=handler)
    runner.session_store._entries[SESSION_KEY] = SimpleNamespace(origin=SessionSource(
        platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", thread_id="42", user_id="u1"))
    return runner


def _watch_event():
    return {
        "type": "watch_match", "event_id": "ev-1", "session_id": "proc_stuck",
        "parent_session_id": "sess_parent", "session_key": SESSION_KEY,
        "pattern": "READY", "command": "build", "output": "READY\n",
    }


def _records(caplog, level, needle):
    return [r for r in caplog.records if r.levelno == level and needle in r.getMessage()]


@pytest.mark.asyncio
async def test_repeated_failure_warns_once_then_debug(monkeypatch, tmp_path, caplog):
    runner = _build_runner(monkeypatch, tmp_path, RefusingHandler())
    completion_queue = queue.Queue()
    completion_queue.put(_watch_event())
    caplog.set_level(logging.DEBUG, logger="gateway.run")

    for _ in range(5):
        await runner._drain_watch_notifications(completion_queue)

    warnings = _records(caplog, logging.WARNING, "notification not delivered")
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for field in ("watch_match", "proc=proc_stuck", "parent_session=sess_parent",
                  f"session_key={SESSION_KEY}", "adapter did not accept the wake"):
        assert field in message
    assert len(_records(caplog, logging.DEBUG, "notification not delivered")) >= 4
    # Still below the cap: requeued, not dropped.
    assert completion_queue.qsize() == 1


@pytest.mark.asyncio
async def test_event_dropped_after_cap(monkeypatch, tmp_path, caplog):
    handler = RefusingHandler()
    runner = _build_runner(monkeypatch, tmp_path, handler)
    completion_queue = queue.Queue()
    completion_queue.put(_watch_event())
    caplog.set_level(logging.DEBUG, logger="gateway.run")

    for _ in range(rn._MAX_DELIVERY_FAILURES - 1):
        await runner._drain_watch_notifications(completion_queue)
    assert completion_queue.qsize() == 1
    assert not _records(caplog, logging.WARNING, "Giving up")

    await runner._drain_watch_notifications(completion_queue)
    assert completion_queue.empty()
    give_up = _records(caplog, logging.WARNING, "Giving up")
    assert len(give_up) == 1
    message = give_up[0].getMessage()
    assert f"after {rn._MAX_DELIVERY_FAILURES} failed delivery attempts" in message
    assert "proc=proc_stuck" in message and "parent_session=sess_parent" in message
    assert "adapter did not accept the wake" in message

    # Nothing left to retry: further drains are no-ops.
    await runner._drain_watch_notifications(completion_queue)
    assert handler.await_count == rn._MAX_DELIVERY_FAILURES


@pytest.mark.asyncio
async def test_successful_delivery_logs_no_failure(monkeypatch, tmp_path, caplog):
    runner = _build_runner(monkeypatch, tmp_path, AdmittingHandler())
    completion_queue = queue.Queue()
    completion_queue.put(_watch_event())
    caplog.set_level(logging.DEBUG, logger="gateway.run")

    await runner._drain_watch_notifications(completion_queue)

    assert completion_queue.empty()
    assert not _records(caplog, logging.WARNING, "not delivered")
    assert not runner._delivery_failure_state()[0]


def test_async_delegation_give_up_drops_ledger_row(monkeypatch, tmp_path):
    import tools.async_delegation as ad
    runner = _build_runner(monkeypatch, tmp_path, RefusingHandler())
    dropped = []
    monkeypatch.setattr(ad, "claim_event_delivery", lambda evt, consumer: "claim-1")
    monkeypatch.setattr(ad, "drop_completion_delivery", lambda did, cid: dropped.append((did, cid)) or True)
    evt = {"type": "async_delegation", "delegation_id": "dlg-1", "session_key": SESSION_KEY}

    results = [runner._count_delivery_failure(evt, "not delivered") for _ in range(rn._MAX_DELIVERY_FAILURES)]

    assert results == [False] * (rn._MAX_DELIVERY_FAILURES - 1) + [True]
    assert dropped == [("dlg-1", "claim-1")]


@pytest.mark.asyncio
async def test_process_watcher_stops_retrying_after_cap(monkeypatch, tmp_path):
    import tools.process_registry as pr_module
    session = SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0, command="sleep 5",
                              started_at=1.0, owner_task_id="", task_id="")

    class _Registry:
        gets = 0

        def get(self, _sid):
            self.gets += 1
            return session if self.gets < 50 else None  # safety stop if the cap regresses

        def is_completion_consumed(self, _sid):
            return False

    monkeypatch.setattr(pr_module, "process_registry", _Registry())

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner = _build_runner(monkeypatch, tmp_path, RefusingHandler())
    runner._enqueue_process_completion_notification = AsyncMock(return_value=False)
    watcher = {"session_id": "proc_sleep", "check_interval": 0, "platform": "telegram", "chat_id": "123",
               "session_key": SESSION_KEY, "notify_on_complete": True}

    await runner._run_process_watcher(watcher)

    assert runner._enqueue_process_completion_notification.await_count == rn._MAX_DELIVERY_FAILURES
