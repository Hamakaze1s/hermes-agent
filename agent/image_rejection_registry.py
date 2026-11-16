"""Process-wide, short-lived record of (provider, model) routes whose tool-result images
cannot be delivered — the route rejected image input (``agent._image_rejecting_models``) or
rejected list-type tool content (``agent._no_list_tool_content_models``).

Those agent sets live on one agent and only steer that agent's send path. Tools
(``vision_analyze``'s native fast path, ``computer_use`` captures) decide whether to hand the
main model a native image envelope without seeing them, so they kept producing pixels the
send path then stripped — leaving the envelope's "you can see it" text to a model with no
image. This registry is the bridge they can read; the agent-side gate
(``VisionMessagePrepMixin._provider_supports_vision_tool_messages``) reads it too, so both
sides decide from the same record.

Entries expire: relays round-robin one model id across several upstream endpoints, so one
endpoint's rejection must not switch the capability off for the life of the process. An agent
that still holds the route in its own (non-expiring) set renews the entry on every send
(``message_sanitization.strip_images_for_rejecting_model`` / ``turn_request_assembly``), so the
tool side cannot reopen the native lane while that agent would still strip the image.

Key scope: ``(provider, model)``, without the base URL. The tool side resolves the main route
from config (``_read_main_provider`` / ``_read_main_model``) and has no reliable view of the
agent's runtime base URL, so a host component would make the two sides miss each other and
reopen the very divergence this module closes. Over-reach is bounded: this is a single-user
VM whose providers each map to one endpoint, and the worst case of a shared key is that
another session on the same provider/model takes the aux-description lane (still vision, one
aux call per image) until the entry expires.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

DEFAULT_TTL_SECONDS = 900.0

_lock = threading.Lock()
_rejections: dict[tuple[str, str], float] = {}  # key -> expiry (monotonic seconds)


def _key(provider: Any, model: Any) -> tuple[str, str] | None:
    if not isinstance(provider, str) or not isinstance(model, str):
        return None
    p, m = provider.strip().lower(), model.strip()
    return (p, m) if p and m else None


def note_rejection(
    provider: Any, model: Any, *, ttl: float = DEFAULT_TTL_SECONDS,
    clock: Optional[Callable[[], float]] = None,
) -> None:
    """Record (or renew) that ``provider``/``model`` can't take tool-result images; no-op for
    empty/non-str ids."""
    key = _key(provider, model)
    if key is None:
        return
    with _lock:
        _rejections[key] = (clock or time.monotonic)() + ttl


def rejects_images(provider: Any, model: Any, *, clock: Optional[Callable[[], float]] = None) -> bool:
    """True while an unexpired rejection is recorded; expired entries are pruned on read."""
    key = _key(provider, model)
    if key is None:
        return False
    now = (clock or time.monotonic)()
    with _lock:
        for stale in [k for k, expiry in _rejections.items() if expiry <= now]:
            del _rejections[stale]
        return key in _rejections


def clear() -> None:
    """Forget every recorded rejection (tests)."""
    with _lock:
        _rejections.clear()
