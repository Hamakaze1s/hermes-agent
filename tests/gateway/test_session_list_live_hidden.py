"""The live session.list picker honours hidden/archived like every stored picker (N110)."""
import inspect
import threading
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection
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
async def test_live_list_reads_state_db_off_the_event_loop(tmp_path, monkeypatch):
    # A per-session get_session on the loop thread starved the liveness watchdog whenever another
    # thread held the GIL (a hung pre_tool_call hook): every sqlite step retakes the GIL. The
    # listing read is the one this handler owns; ``authority.resolve`` keeps its own reads on the
    # loop (pre-existing, unchanged here), so the assertion is scoped to the listing call.
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    loop_thread = threading.get_ident()
    readers = []
    real_get_session = authority.db.get_session

    def recording_get_session(session_id):
        frames = [frame.function for frame in inspect.stack()[1:8]]
        readers.append(('_live_rows' in frames, threading.get_ident()))
        return real_get_session(session_id)

    try:
        chat = await _create(conn, authority, store, 'cli', 'chat')
        monkeypatch.setattr(authority.db, 'get_session', recording_get_session)
        listed = await conn.dispatch({'id': 11, 'method': 'session.list', 'params': {}})
        assert [row['id'] for row in listed['result']['sessions']] == [chat]
        listing_reads = [thread for from_live_rows, thread in readers if from_live_rows]
        assert listing_reads, 'the live listing must read the session row'
        assert loop_thread not in listing_reads
    finally:
        await conn.close()
        authority.db.close()
