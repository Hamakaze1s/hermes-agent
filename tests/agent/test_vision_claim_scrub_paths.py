"""Every path that removes a tool image must also remove the native envelope's sight claim.

The native ``vision_analyze`` envelope says "Image loaded into your context — you can see it
natively now. Use your built-in vision…". Once a stripper deletes the image part, that text
is a lie the model acts on. The fixtures are built from the REAL envelope builder, so a
rewording that the scrub patterns stop matching turns these tests red.
"""
from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import pytest

from agent.vision_claim_text import (
    IMAGE_OMITTED_MARKER, has_native_vision_claim, scrub_native_vision_claims,
)
from tools.vision_tools import _build_native_vision_tool_result

_DATA_URL = "data:image/png;base64,iVBORw0KGgo="


def _envelope(question="what does the chart say?", scale_note=None):
    return _build_native_vision_tool_result(
        "/tmp/chart.png", question, _DATA_URL, 2048, scale_note=scale_note,
    )


def _envelope_text() -> str:
    return next(p["text"] for p in _envelope()["content"] if p.get("type") == "text")


def _tool_msg(content):
    return {"role": "tool", "tool_call_id": "call_1", "content": content}


def _all_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        p if isinstance(p, str) else str(p.get("text", ""))
        for p in content if isinstance(p, (str, dict))
    )


def _assert_honest(content):
    text = _all_text(content)
    assert not has_native_vision_claim(text), text
    for phrase in ("you can see it", "built-in vision"):
        assert phrase not in text, text
    assert IMAGE_OMITTED_MARKER in text, text
    if isinstance(content, list):
        assert not any(isinstance(p, dict) and p.get("type") in {"image_url", "input_image", "image"}
                       for p in content)


# ─── scrub ↔ real envelope wording ───────────────────────────────────────────


class TestScrubBindsToRealEnvelope:

    def test_patterns_hit_the_real_envelope_text(self):
        """Reword the envelope without updating the patterns → this goes red."""
        text = _envelope_text()
        assert has_native_vision_claim(text)
        scrubbed = scrub_native_vision_claims(text)
        assert scrubbed != text
        assert not has_native_vision_claim(scrubbed)
        assert "you can see it" not in scrubbed and "built-in vision" not in scrubbed

    def test_question_and_note_survive(self):
        text = next(p["text"] for p in _envelope(scale_note="downscaled 50%")["content"]
                    if p.get("type") == "text")
        scrubbed = scrub_native_vision_claims(text)
        assert scrubbed.startswith(IMAGE_OMITTED_MARKER)
        assert "Question: what does the chart say?" in scrubbed
        assert "Note: downscaled 50%" in scrubbed

    def test_idempotent(self):
        once = scrub_native_vision_claims(_envelope_text())
        assert scrub_native_vision_claims(once) == once
        assert once.count(IMAGE_OMITTED_MARKER) == 1

    def test_unrelated_text_untouched(self):
        assert scrub_native_vision_claims("plain tool output") == "plain tool output"

    def test_private_aliases_still_exported(self):
        from agent import vision_message_prep as vmp
        assert vmp._scrub_native_vision_claims is scrub_native_vision_claims
        assert vmp._IMAGE_OMITTED_MARKER == IMAGE_OMITTED_MARKER
        assert vmp._NATIVE_VISION_CLAIM_PATTERNS


# ─── message_sanitization._strip_images_from_messages (send-path strip) ─────


class TestSanitizationStrip:

    def test_native_envelope_parts_are_scrubbed(self):
        from agent.message_sanitization import _strip_images_from_messages

        msgs = [_tool_msg(copy.deepcopy(_envelope()["content"]))]
        assert _strip_images_from_messages(msgs) is True
        _assert_honest(msgs[0]["content"])
        assert "Question: what does the chart say?" in _all_text(msgs[0]["content"])

    def test_strip_does_not_mutate_shared_text_part(self):
        from agent.message_sanitization import _strip_images_from_messages

        parts = copy.deepcopy(_envelope()["content"])
        text_part = parts[0]
        original = text_part["text"]
        _strip_images_from_messages([_tool_msg(parts)])
        assert text_part["text"] == original

    def test_rejecting_model_send_path(self):
        from agent.message_sanitization import strip_images_for_rejecting_model

        agent = SimpleNamespace(provider="opencode-go", model="deepseek-v4.1-flash",
                                _image_rejecting_models={("opencode-go", "deepseek-v4.1-flash")})
        msgs = [_tool_msg(copy.deepcopy(_envelope()["content"]))]
        assert strip_images_for_rejecting_model(agent, msgs) is True
        _assert_honest(msgs[0]["content"])


# ─── context_compressor strippers (history budget / compaction) ─────────────


class TestCompressorStrip:

    def test_strip_images_from_tool_msg_list_content(self):
        from agent.context_compressor import _strip_images_from_tool_msg

        new_msg = _strip_images_from_tool_msg(_tool_msg(copy.deepcopy(_envelope()["content"])))
        assert new_msg is not None
        _assert_honest(new_msg["content"])

    def test_strip_images_from_tool_msg_envelope_summary(self):
        """The ``_multimodal`` envelope collapses to its text_summary — which must not claim sight
        either (scrubbed if a producer ever puts the claim there)."""
        from agent.context_compressor import _strip_images_from_tool_msg

        env = _envelope()
        new_msg = _strip_images_from_tool_msg(_tool_msg(env))
        assert not has_native_vision_claim(new_msg["content"])

        env_with_claim = {**env, "text_summary": _envelope_text()}
        new_msg = _strip_images_from_tool_msg(_tool_msg(env_with_claim))
        _assert_honest(new_msg["content"])

    def test_strip_images_from_content(self):
        from agent.context_compressor import _strip_images_from_content

        content = copy.deepcopy(_envelope()["content"])
        out = _strip_images_from_content(content)
        assert out is not content
        _assert_honest(out)
        # Input untouched (compaction builds new rows).
        assert has_native_vision_claim(content[0]["text"])

    def test_strip_twice_is_stable(self):
        from agent.context_compressor import _strip_images_from_content

        once = _strip_images_from_content(copy.deepcopy(_envelope()["content"]))
        assert _strip_images_from_content(once) is once  # no images left → unchanged


# ─── 422 list-content strip (already scrubbed; guard against regressions) ───


def test_list_content_strip_still_scrubs():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.provider, agent.model = "opencode-go", "deepseek-v4.1-flash"
    msgs = [_tool_msg(copy.deepcopy(_envelope()["content"]))]
    assert agent._try_strip_image_parts_from_tool_messages(msgs, remember_model=False) is True
    _assert_honest(msgs[0]["content"])


# ─── send-path pre-strip for models known to reject list-type tool content ──


def _assembly_agent(no_list_models):
    from agent.vision_message_prep import VisionMessagePrepMixin

    agent = SimpleNamespace(
        api_mode="chat_completions", provider="opencode-go", model="deepseek-v4.1-flash",
        base_url="https://opencode.ai/zen/go/v1", session_id="s1", _session_db=None,
        _persist_disabled=True, _current_turn_timestamp=1.0, ephemeral_system_prompt="",
        prefill_messages=[], tools=[], _use_prompt_caching=False, _usage_anchor=None,
        verbose_logging=False, reasoning_callback=None, stream_delta_callback=None,
        _stream_callback=None, log_prefix="", context_compressor=None,
        _no_list_tool_content_models=set(no_list_models), _image_rejecting_models=set(),
    )
    agent._needs_thinking_reasoning_pad = lambda: False
    agent._should_sanitize_tool_calls = lambda: False
    agent._sanitize_api_messages = lambda value: value
    agent._emit_warning = agent._vprint = lambda *args, **kwargs: None
    agent._drop_thinking_only_and_merge_users = lambda value, **kwargs: value
    agent._copy_reasoning_content_for_api = lambda source, target: None
    agent._content_has_image_parts = VisionMessagePrepMixin._content_has_image_parts
    agent._try_strip_image_parts_from_tool_messages = (
        lambda msgs, **kw: VisionMessagePrepMixin._try_strip_image_parts_from_tool_messages(agent, msgs, **kw)
    )
    return agent


def _history():
    return [
        {"role": "user", "content": "look at the chart"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "vision_analyze", "arguments": "{}"}}]},
        _tool_msg(copy.deepcopy(_envelope()["content"])),
        {"role": "user", "content": "well?"},
    ]


def _assemble(agent, history, monkeypatch):
    import agent.conversation_loop as loop
    from agent.turn_request_assembly import assemble_api_request

    monkeypatch.setattr(loop, "_apply_context_engine_selection",
                        lambda agent, api, history, incoming, logger=None: api)
    monkeypatch.setattr(loop, "_midturn_request_pressure_tokens",
                        lambda agent, messages, system, approx: approx)
    monkeypatch.setattr(loop, "_pressure_with_real_floor", lambda compressor, value: value)
    return assemble_api_request(
        agent, messages=history, current_turn_user_idx=len(history) - 1,
        _ext_prefetch_cache="", _plugin_user_context="", moa_config=None,
        active_system_prompt="", original_user_message=history[-1]["content"],
        pending_moa_prepared_request=None, request_logger=logging.getLogger("vision-claim-test"),
    )


def _tool_rows(api_messages):
    return [m for m in api_messages if m.get("role") == "tool"]


def test_known_list_rejecting_model_strips_request_copy_only(monkeypatch):
    agent = _assembly_agent({("opencode-go", "deepseek-v4.1-flash")})
    history = _history()
    before = copy.deepcopy(history)

    assembled = _assemble(agent, history, monkeypatch)

    [tool] = _tool_rows(assembled.api_messages)
    assert isinstance(tool["content"], str)
    _assert_honest(tool["content"])
    assert history == before  # session history keeps its image
    # remember_model=False: nothing new recorded by the pre-strip.
    assert agent._no_list_tool_content_models == {("opencode-go", "deepseek-v4.1-flash")}


def _many_envelope_history(n):
    from agent.image_eviction_policy import OUTBOUND_IMAGE_LIMIT

    assert n > OUTBOUND_IMAGE_LIMIT  # enough carriers that eviction retires some
    history = [{"role": "user", "content": "look at the charts"}]
    for i in range(n):
        history.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": "vision_analyze", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": f"call_{i}",
                        "content": copy.deepcopy(_envelope()["content"])})
    history.append({"role": "user", "content": "well?"})
    return history


def test_pre_strip_runs_before_eviction_so_no_tool_list_survives(monkeypatch):
    """Eviction rewrites a stale tool image into a text-only LIST; the list-content pre-strip
    skips lists without images. Pre-strip first, or the known-bad endpoint still 400s."""
    from agent.context_compressor import evict_stale_outbound_tool_images

    # The hazard is real: eviction alone leaves list-type tool content behind.
    alone = _many_envelope_history(30)
    assert evict_stale_outbound_tool_images(alone) > 0
    assert any(isinstance(m["content"], list) and not any(p.get("type") == "image_url" for p in m["content"])
               for m in _tool_rows(alone))

    agent = _assembly_agent({("opencode-go", "deepseek-v4.1-flash")})
    assembled = _assemble(agent, _many_envelope_history(30), monkeypatch)
    rows = _tool_rows(assembled.api_messages)
    assert len(rows) == 30
    for row in rows:
        assert isinstance(row["content"], str), row["content"]
        assert not has_native_vision_claim(row["content"])


def test_list_rejecting_assembly_renews_registry(monkeypatch):
    from agent.image_rejection_registry import rejects_images

    agent = _assembly_agent({("opencode-go", "deepseek-v4.1-flash")})
    _assemble(agent, _history(), monkeypatch)
    assert rejects_images("opencode-go", "deepseek-v4.1-flash") is True


# ─── M3: "already in context" short-circuit ─────────────────────────────────


def _png(path):
    from PIL import Image

    Image.new("RGB", (16, 16), (200, 30, 30)).save(path)
    return str(path)


def test_already_in_context_reply_is_conditional(tmp_path):
    from agent.image_routing import build_native_content_parts
    from tools import vision_tools_history_budget as budget

    path = _png(tmp_path / "shot.png")
    parts, skipped = build_native_content_parts("look", [path])
    assert not skipped
    with budget.native_turn_images(parts):
        reply = budget.native_turn_duplicate(path, None)
    assert reply is not None
    import json

    message = json.loads(reply)["message"]
    for phrase in ("you can see it", "built-in vision", "can see it now"):
        assert phrase not in message
    assert not has_native_vision_claim(message)
    assert "If no image is visible there, it did not reach you" in message


@pytest.mark.asyncio
async def test_already_in_context_short_circuit_skipped_for_rejecting_route(tmp_path, monkeypatch):
    import json

    from agent.image_rejection_registry import note_rejection
    from agent.image_routing import build_native_content_parts
    from tools import vision_tools, vision_tools_history_budget as budget

    monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: "opencode-go")
    monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: "deepseek-v4.1-flash")
    path = _png(tmp_path / "shot.png")
    parts, _ = build_native_content_parts("look", [path])
    with budget.native_turn_images(parts):
        first = await vision_tools._vision_analyze_native(path, "q")
        assert json.loads(first)["already_in_context"] is True
        note_rejection("opencode-go", "deepseek-v4.1-flash")
        second = await vision_tools._vision_analyze_native(path, "q")
    # No "it's already attached" claim: the request goes on to load the image (a lane the
    # agent side then strips or describes) instead of pointing at a possibly stripped image.
    assert isinstance(second, dict) and second.get("_multimodal") is True


# ─── unsupported-format branches never see a native envelope ────────────────


@pytest.mark.asyncio
async def test_native_envelope_images_are_always_a_supported_format(tmp_path):
    """Why anthropic_message_convert / codex_responses_adapter leave sibling text alone on the
    "[image omitted: … not a supported image format]" branch: the envelope's image is
    normalized before encoding, so that branch can't receive it."""
    from PIL import Image

    from tools import vision_tools
    from tools.vision_tools_image_prep import unsupported_inline_image_media_type

    bmp = tmp_path / "shot.bmp"
    Image.new("RGB", (16, 16), (10, 120, 200)).save(bmp, format="BMP")
    env = await vision_tools._vision_analyze_native(str(bmp), "q")
    assert isinstance(env, dict), env
    [url] = [p["image_url"]["url"] for p in env["content"] if p.get("type") == "image_url"]
    assert url.startswith("data:image/")
    assert unsupported_inline_image_media_type(url) is None


# ─── /model → text-only model: description + scrubbed text ──────────────────


def test_non_vision_model_conversion_scrubs_envelope_text(monkeypatch):
    import json

    from agent.vision_claim_text import IMAGE_DESCRIBED_MARKER
    from run_agent import AIAgent

    async def _aux(image_url, user_prompt, model=None, **kw):
        return json.dumps({"success": True, "analysis": "A red square."})

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _aux)
    agent = object.__new__(AIAgent)
    agent.provider, agent.model = "opencode-go", "glm-5.3-flash"
    agent._anthropic_image_fallback_cache = {}
    monkeypatch.setattr(AIAgent, "_model_supports_vision", lambda self: False)

    [out] = agent._prepare_messages_for_non_vision_model([_tool_msg(copy.deepcopy(_envelope()["content"]))])
    text = out["content"]
    assert isinstance(text, str)
    assert "A red square." in text
    assert not has_native_vision_claim(text)
    assert IMAGE_DESCRIBED_MARKER in text and IMAGE_OMITTED_MARKER not in text
    assert "Question: what does the chart say?" in text


# ─── Anthropic wire eviction ────────────────────────────────────────────────


def test_anthropic_eviction_scrubs_sibling_text():
    from agent.anthropic_message_convert import _content_parts_to_anthropic_blocks, _evict_old_screenshots
    from agent.image_eviction_policy import OUTBOUND_IMAGE_LIMIT

    inner = _content_parts_to_anthropic_blocks(_envelope()["content"])
    assert any(b.get("type") == "image" for b in inner) and has_native_vision_claim(inner[0]["text"])
    result = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": f"t{i}", "content": copy.deepcopy(inner)}
        for i in range(OUTBOUND_IMAGE_LIMIT + 5)
    ]}]
    _evict_old_screenshots(result)

    blocks = result[0]["content"]
    retired = [b for b in blocks if not any(x.get("type") == "image" for x in b["content"])]
    kept = [b for b in blocks if any(x.get("type") == "image" for x in b["content"])]
    assert retired and kept
    for block in retired:
        text = _all_text(block["content"])
        assert not has_native_vision_claim(text) and IMAGE_OMITTED_MARKER in text
    for block in kept:  # images still attached: the claim is true and stays
        assert has_native_vision_claim(_all_text(block["content"]))


# ─── MoA reference view ─────────────────────────────────────────────────────


def test_moa_reference_view_scrubs_flattened_tool_result():
    from agent.moa_loop import _reference_messages

    rendered = _reference_messages(_history())
    text = "\n".join(m["content"] for m in rendered)
    assert "[tool result:" in text
    assert not has_native_vision_claim(text)
    assert "you can see it" not in text and "built-in vision" not in text


def test_unflagged_model_keeps_tool_image(monkeypatch):
    agent = _assembly_agent(set())
    history = _history()

    assembled = _assemble(agent, history, monkeypatch)

    [tool] = _tool_rows(assembled.api_messages)
    assert isinstance(tool["content"], list)
    assert any(p.get("type") == "image_url" for p in tool["content"])
    assert agent._no_list_tool_content_models == set()
