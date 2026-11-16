"""Scrub "you can see the image" claims from text that outlived its image.

The native-vision envelope (``tools.vision_tools._build_native_vision_tool_result``) tells
the model to read the attached image with its own eyes. Every path that later removes the
image bytes — the 4xx list-content strip, the send-path strip for an image-rejecting model,
history-budget / compression stripping — must also drop that sentence, or the model
describes a picture it can no longer see (#132837 follow-up). One module, so the phrasing
and the honest marker cannot drift between strippers.
"""
from __future__ import annotations

import re
from typing import Any

# Whole sentences, not fragments: removing only "you can see it natively now" leaves
# "Image loaded into your context — . Use your built-in vision…", which still claims a
# delivery that no longer exists. No marker may suggest calling vision_analyze again —
# while the native fast path is open that would re-enter the same rejection.
NATIVE_VISION_CLAIM_PATTERNS = (
    re.compile(r"Image loaded into your context\s*[—–-]?\s*you can see it natively now\."),
    re.compile(r"Use your built-in vision to answer the user\."),
)
IMAGE_OMITTED_MARKER = (
    "[the image was removed from this tool result — it is not visible to you and no "
    "description is available.]"
)
# For text that sits next to an aux-vision description of the removed image: the loss is
# real, but "no description is available" would contradict the description beside it.
IMAGE_DESCRIBED_MARKER = (
    "[the image itself is not attached — it is not visible to you; rely only on the "
    "description above, produced by a separate vision model.]"
)

_TEXT_PART_TYPES = frozenset({"text", "input_text"})


def has_native_vision_claim(text: Any) -> bool:
    return isinstance(text, str) and any(p.search(text) for p in NATIVE_VISION_CLAIM_PATTERNS)


def scrub_native_vision_claims(text: str, marker: str = IMAGE_OMITTED_MARKER) -> str:
    """Drop "you can see it" sentences from salvaged tool text and mark the loss honestly.

    Only the claim sentences are removed: the envelope's trailing ``Question:`` / ``Note:``
    blocks keep their line structure and their own punctuation. Idempotent — the output
    carries no claim, so a second pass returns it unchanged.
    """
    if not has_native_vision_claim(text):
        return text
    for pattern in NATIVE_VISION_CLAIM_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return "\n\n".join(part for part in (marker, text) if part)


def scrub_native_vision_claims_in_parts(parts: list, marker: str = IMAGE_OMITTED_MARKER) -> list:
    """New list: ``parts`` with every text part (str or text-typed dict) scrubbed. Dicts are
    copied only when rewritten, so parts shared with history are never mutated. A scrubbed
    part is never empty (it carries the marker), so no part disappears."""
    out: list = []
    for part in parts:
        if isinstance(part, str):
            part = scrub_native_vision_claims(part, marker)
        elif isinstance(part, dict) and part.get("type") in _TEXT_PART_TYPES:
            text = part.get("text")
            if has_native_vision_claim(text):
                part = {**part, "text": scrub_native_vision_claims(text, marker)}
        out.append(part)
    return out
