"""Regression tests for the delete/prune vs. live session.list / resolve race.

Deletion and prune commit the state.db row removal off-loop before retiring the
live entry from ``authority.sessions``. Both ``SessionAuthority._handle`` (called by
``resolve`` on the event loop) and ``AuthorityConnection.list_sessions._live_rows``
(called in ``asyncio.to_thread`` after ``resolve``) must treat a missing row as gone
rather than crashing with ``TypeError`` or ``AttributeError``.
"""
import threading
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_local_source_labels import _authority, _create


@pytest.mark.asyncio
async def test_resolve_raises_not_found_when_row_deleted_before_retire(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        survivor = await _create(conn, authority, store, 'cli', 'survivor')
        doomed = await _create(conn, authority, store, 'cli', 'doomed')

        # Simulate the window after DB delete commits and before retire_live_sessions pops it.
        assert authority.db.delete_session(doomed) is True
        assert doomed in authority.sessions

        with pytest.raises(RuntimeStoreError) as exc_info:
            await authority.resolve(conn.actor, SessionRef(conn.actor.profile_id, doomed))
        assert exc_info.value.reason == 'not_found'

        info = await conn.dispatch({'id': 1, 'method': 'session.info', 'params': {'session_id': doomed}})
        assert info['error']['code'] == 4001
        assert info['error']['data']['reason'] == 'not_found'

        listed = await conn.dispatch({'id': 2, 'method': 'session.list', 'params': {}})
        assert [row['id'] for row in listed['result']['sessions']] == [survivor]
    finally:
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_live_list_skips_session_deleted_between_resolve_and_worker_read(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    loop_thread = threading.get_ident()
    real_get_session = authority.db.get_session
    deleted_in_window = False

    try:
        survivor = await _create(conn, authority, store, 'cli', 'survivor')
        doomed = await _create(conn, authority, store, 'cli', 'doomed')

        def delete_before_worker_read(session_id):
            nonlocal deleted_in_window
            if (not deleted_in_window
                    and session_id == doomed
                    and threading.get_ident() != loop_thread):
                deleted_in_window = True
                assert authority.db.delete_session(doomed) is True
            return real_get_session(session_id)

        monkeypatch.setattr(authority.db, 'get_session', delete_before_worker_read)
        listed = await conn.dispatch({'id': 3, 'method': 'session.list', 'params': {}})
        assert deleted_in_window is True
        assert [row['id'] for row in listed['result']['sessions']] == [survivor]
    finally:
        await conn.close()
        authority.db.close()
