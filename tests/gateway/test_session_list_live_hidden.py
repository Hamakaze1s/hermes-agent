"""The live session.list picker honours hidden/archived like every stored picker (N110), and
answers from batched projections instead of one thick session row per session (P25)."""
import threading
from types import SimpleNamespace

import pytest

import hermes_state_local
import hermes_state_runtime
from gateway.session_controls import AuthorityConnection
from gateway.session_contract import SessionRef
from gateway.session_local_close import closed_and_unviewed
from hermes_state_health import storage_state
from hermes_state_local import POLICY_PREFIX, local_targets
from hermes_state_runtime import session_summaries
from tests.gateway.test_local_source_labels import _authority, _create


@pytest.mark.asyncio
async def test_live_list_hides_hidden_and_archived_unless_asked(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        chat = await _create(conn, authority, store, 'cli', 'chat')
        hidden = await _create(conn, authority, store, 'cli', 'hidden')
        archived = await _create(conn, authority, store, 'cli', 'archived')
        authority.db.set_session_hidden(hidden, True)
        authority.db.set_session_archived(archived, True)
        listed = await conn.dispatch({'id': 9, 'method': 'session.list', 'params': {}})
        assert [row['id'] for row in listed['result']['sessions']] == [chat]
        every = await conn.dispatch({'id': 10, 'method': 'session.list', 'params': {'include_hidden': True}})
        assert {row['id'] for row in every['result']['sessions']} == {chat, hidden, archived}
    finally:
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_live_list_batches_reads_and_never_loads_the_thick_session_row(tmp_path, monkeypatch):
    # Why this exists: every TUI frontend polls session.list every 1.5 s, and the gateway restores
    # the whole local roster at boot, so a per-session `get_session` here (which also materializes
    # the resolved system prompt and tool manifest, ~57 KB a row on a real store) made this handler
    # the gateway's largest CPU consumer and starved its event loop. The listing must answer from
    # batched summary reads, off the loop thread.
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    loop_thread = threading.get_ident()
    summary_calls = []
    real_summaries = hermes_state_runtime.session_summaries

    def recording_summaries(db, *, session_ids):
        summary_calls.append(threading.get_ident())
        return real_summaries(db, session_ids=session_ids)

    def forbidden_get_session(session_id):
        raise AssertionError('the live listing must not read a thick session row')

    try:
        created = [await _create(conn, authority, store, 'cli', f'chat-{i}') for i in range(12)]
        monkeypatch.setattr(hermes_state_runtime, 'session_summaries', recording_summaries)
        monkeypatch.setattr(authority.db, 'get_session', forbidden_get_session)
        listed = await conn.dispatch({'id': 11, 'method': 'session.list', 'params': {}})
        assert {row['id'] for row in listed['result']['sessions']} == set(created)
        assert summary_calls, 'the live listing must read the session summaries'
        assert loop_thread not in summary_calls, 'the listing read must stay off the event loop'
    finally:
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_live_list_matches_the_handle_path_on_a_closed_session(tmp_path, monkeypatch):
    # The batched listing must reproduce the per-session handle path exactly: a live entry whose
    # row is ended and whose last viewer left is listed by the STORED feed, not as live; while a
    # viewer is still attached the same row stays in the live feed.
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        closed = await _create(conn, authority, store, 'cli', 'closed')
        authority.db.end_session(closed, 'tui_close')
        ref = SessionRef(conn.actor.profile_id, closed)
        handle = authority._handle(ref)
        summaries = session_summaries(authority.db, session_ids=[closed])
        states = authority.pending_states([closed])
        targets = local_targets(authority.db, [closed])

        def batched():
            return closed_and_unviewed(authority, closed, state=states[closed],
                                       rows=summaries, targets=targets)

        assert authority.sessions[closed].subscribers, 'session.create attaches its creator'
        assert closed_and_unviewed(authority, closed, handle) is False
        assert batched() is False
        listed = await conn.dispatch({'id': 12, 'method': 'session.list', 'params': {}})
        assert [row['id'] for row in listed['result']['sessions']] == [closed]

        authority.sessions[closed].subscribers.clear()  # the viewer went away
        assert closed_and_unviewed(authority, closed, handle) is True
        assert batched() is True
        listed = await conn.dispatch({'id': 13, 'method': 'session.list', 'params': {}})
        assert listed['result']['sessions'] == []
    finally:
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_live_list_keeps_an_ended_session_without_a_local_receipt(tmp_path, monkeypatch):
    # Routed (telegram/API) live entries carry no gateway.local_policy receipt. The handle path
    # has always answered "not closed" without one, so an ended, idle, unviewed routed session
    # stays in the live feed; the batched path must not start hiding it.
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        routed = await _create(conn, authority, store, 'cli', 'routed')
        authority.db.end_session(routed, 'tui_close')
        authority.sessions[routed].subscribers.clear()
        for targets in ({}, {routed: None}, {routed: ''}):
            monkeypatch.setattr(hermes_state_local, 'local_targets', lambda db, ids, _t=targets: dict(_t))
            listed = await conn.dispatch({'id': 14, 'method': 'session.list', 'params': {}})
            assert [row['id'] for row in listed['result']['sessions']] == [routed], targets
    finally:
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_local_targets_tolerates_a_malformed_receipt(tmp_path, monkeypatch):
    # json_extract raises "malformed JSON" on a bad receipt, and the read helper's storage hook
    # classifies that as structural corruption and latches the whole store for the process. One
    # bad row must read as "no receipt" instead (what local_receipt's json.loads guard did).
    authority, store = await _authority(tmp_path, monkeypatch)
    try:
        good = await _create(AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None),
                                                 {'user_id': 'owner'}), authority, store, 'cli', 'good')
        authority.db._write_sql('INSERT OR REPLACE INTO state_meta(key, value) VALUES(?,?)',
                                (POLICY_PREFIX + 'broken', '{not json'))
        targets = local_targets(authority.db, ['broken', good])
        # A bad receipt reads as "no target" (falsy), exactly like local_receipt's None arm.
        assert not targets.get('broken')
        assert targets.get(good) == good
        assert storage_state(authority.db.db_path) == 'ok'
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_live_list_keeps_a_resumable_session_while_a_claim_is_pending(tmp_path, monkeypatch):
    # An 'unknown' admission claim is neither running nor "closed": the row must stay listed even
    # when it is ended and has no viewer (the state guard, not the ended_at check, decides).
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        sid = await _create(conn, authority, store, 'cli', 'paused')
        authority.db.end_session(sid, 'tui_close')
        authority.sessions[sid].subscribers.clear()
        real_states = hermes_state_runtime.pending_admission_states
        monkeypatch.setattr(hermes_state_runtime, 'pending_admission_states',
                            lambda db, *, session_ids: {**real_states(db, session_ids=session_ids),
                                                        sid: {'unknown'}})
        listed = await conn.dispatch({'id': 16, 'method': 'session.list', 'params': {}})
        assert [row['id'] for row in listed['result']['sessions']] == [sid]
    finally:
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_closed_and_unviewed_follows_the_receipt_target_row(tmp_path, monkeypatch):
    # A local receipt may name a different physical row (compression tip). The batched branch must
    # read THAT row's ended_at, in both directions.
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        own = await _create(conn, authority, store, 'cli', 'own')
        target = await _create(conn, authority, store, 'cli', 'target')
        authority.sessions[own].subscribers.clear()
        authority.db.end_session(target, 'tui_close')

        def batched():
            return closed_and_unviewed(authority, own, state='idle',
                                       rows=session_summaries(authority.db, session_ids=[own]),
                                       targets={own: target})

        assert batched() is True   # the target row is ended -> main lists it as stored
        authority.db.reopen_session(target)
        authority.db.end_session(own, 'tui_close')
        assert batched() is False  # the own row is ended but the receipt's target is not
    finally:
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_live_list_marks_running_from_a_started_admission(tmp_path, monkeypatch):
    # The listing's `running` flag comes from the batched admission states: a started claim is
    # running, an unknown claim is neither running nor "closed", and neither hides the row.
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        busy = await _create(conn, authority, store, 'cli', 'busy')
        unknown = await _create(conn, authority, store, 'cli', 'unknown')
        real_states = hermes_state_runtime.pending_admission_states

        def fake_states(db, *, session_ids):
            raw = real_states(db, session_ids=session_ids)
            raw.setdefault(busy, set()).add('started')
            raw.setdefault(unknown, set()).add('unknown')
            return raw

        monkeypatch.setattr(hermes_state_runtime, 'pending_admission_states', fake_states)
        listed = await conn.dispatch({'id': 15, 'method': 'session.list', 'params': {}})
        by_id = {row['id']: row for row in listed['result']['sessions']}
        assert by_id[busy]['running'] is True
        assert by_id[unknown]['running'] is False
    finally:
        await conn.close()
        authority.db.close()
