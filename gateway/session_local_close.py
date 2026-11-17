"""The picker's view of a canonical local session its last viewer closed, as main listed it.

Main's sidecar stamped ``tui_close`` (``/new``, a resume switch) and ``tui_shutdown`` (exit) on the
session row it owned; pickers then listed it as resumable. The stamp itself is the owner's
``session.detach {end_reason}`` (``session_acp_lifecycle.end_idle_viewer_session``); this module only
decides which live rows read as closed.
"""
from hermes_state_local import local_receipt
from hermes_state_runtime import RuntimeStoreError


def _local_target(authority, session_id):
    try:
        return local_receipt(authority.db, session_id)['entry']['session_id']
    except (RuntimeStoreError, KeyError, TypeError):
        return None  # not a local session (API binding, malformed receipt): nothing to read as closed


def closed_and_unviewed(authority, session_id, handle=None, *, state=None, rows=None, targets=None):
    """A live-feed row main would list as resumable: closed, idle and with no viewer left.

    ``state`` is the batched substitute for ``handle.execution_state``; ``rows``/``targets`` let a
    listing pass the summaries and receipt targets it already loaded instead of re-reading
    state.db once per row (one live listing walks the whole recovered local roster).
    """
    live = authority.sessions.get(session_id)
    if state is not None:
        execution_state = state
    elif handle is not None:
        execution_state = handle.execution_state
    else:
        raise TypeError('closed_and_unviewed needs a session handle or an execution state')
    if live is None or live.subscribers or execution_state != 'idle':
        return False
    target = targets.get(session_id) if targets is not None else _local_target(authority, session_id)
    if not target:
        # No receipt / unreadable receipt: not a local session, so nothing to read as closed.
        # A routed (telegram/API) live entry must keep listing even when its row is ended — the
        # handle path has always answered False here and the batched path must match it.
        return False
    if rows is not None and target == session_id:
        row = rows.get(session_id)
    elif rows is not None:
        from hermes_state_runtime import session_summaries
        row = session_summaries(authority.db, session_ids=(target,)).get(target)
    else:
        row = authority.db.get_session(target)
    return bool(row and row.get('ended_at') is not None)
