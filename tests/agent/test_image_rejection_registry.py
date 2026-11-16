"""A route that rejected image input this session must close the tool-side native lane.

Without the process-wide registry, ``vision_analyze`` kept handing the main model a native
envelope ("you can see it natively now") that the send path then stripped. Entries expire
(900 s): the opencode-go relay round-robins one id over several upstream endpoints, so one
endpoint's rejection must not disable the lane for good.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import image_rejection_registry as registry

PROVIDER, MODEL = "opencode-go", "deepseek-v4.1-flash"
# Deterministic route: native image mode + declared vision, so only the tool-message gate decides.
_CFG = {"agent": {"image_input_mode": "native"}, "model": {"supports_vision": True}}


@pytest.fixture(autouse=True)
def _fresh_registry():
    import providers

    providers.list_providers()
    registry.clear()
    yield
    registry.clear()


class _Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


# ─── registry ────────────────────────────────────────────────────────────────


def test_note_then_rejects():
    assert registry.rejects_images(PROVIDER, MODEL) is False
    registry.note_rejection(PROVIDER, MODEL)
    assert registry.rejects_images(PROVIDER, MODEL) is True


def test_provider_case_normalized_model_exact():
    registry.note_rejection("OpenCode-Go ", MODEL)
    assert registry.rejects_images("opencode-go", MODEL) is True
    assert registry.rejects_images("opencode-go", "DeepSeek-V4.1-Flash") is False
    assert registry.rejects_images("opencode-go", "glm-5.3-flash") is False


def test_default_ttl_is_900_seconds():
    assert registry.DEFAULT_TTL_SECONDS == 900


def test_entry_expires_after_ttl():
    clock = _Clock()
    registry.note_rejection(PROVIDER, MODEL, clock=clock)
    clock.now += 899
    assert registry.rejects_images(PROVIDER, MODEL, clock=clock) is True
    clock.now += 2
    assert registry.rejects_images(PROVIDER, MODEL, clock=clock) is False
    # Pruned on read, not merely hidden.
    assert registry._rejections == {}


def test_expiry_via_monotonic_patch(monkeypatch):
    registry.note_rejection(PROVIDER, MODEL)
    real = registry.time.monotonic()
    monkeypatch.setattr(registry.time, "monotonic", lambda: real + registry.DEFAULT_TTL_SECONDS + 1)
    assert registry.rejects_images(PROVIDER, MODEL) is False


@pytest.mark.parametrize("provider, model", [
    ("", MODEL), (PROVIDER, ""), ("  ", "  "), (None, MODEL), (PROVIDER, None),
    (None, None), (1, MODEL), (PROVIDER, ["x"]),
])
def test_bad_ids_never_reject_and_are_not_recorded(provider, model):
    registry.note_rejection(provider, model)
    assert registry.rejects_images(provider, model) is False
    assert registry._rejections == {}


# ─── tool-side gate (tools.vision_tools) ─────────────────────────────────────


def test_native_tool_images_open_for_verified_route():
    from tools.vision_tools import _native_tool_result_images

    assert _native_tool_result_images(PROVIDER, MODEL, _CFG) is True


def test_registry_hit_closes_native_tool_images():
    from tools.vision_tools import _accepts_tool_result_images, _native_tool_result_images

    registry.note_rejection(PROVIDER, MODEL)
    assert _accepts_tool_result_images(PROVIDER, MODEL, _CFG) is False
    assert _native_tool_result_images(PROVIDER, MODEL, _CFG) is False


def test_registry_hit_expires_back_to_native(monkeypatch):
    from tools.vision_tools import _native_tool_result_images

    registry.note_rejection(PROVIDER, MODEL)
    real = registry.time.monotonic()
    monkeypatch.setattr(registry.time, "monotonic", lambda: real + registry.DEFAULT_TTL_SECONDS + 1)
    assert _native_tool_result_images(PROVIDER, MODEL, _CFG) is True


@pytest.mark.asyncio
async def test_vision_analyze_falls_back_to_aux_user_message_lane(monkeypatch):
    """Hard constraint: a registry hit may only downgrade to the aux lane (image in a USER
    message) — never to no vision. The handler must reach vision_analyze_tool with the image."""
    import tools.vision_tools as vt

    monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: PROVIDER)
    monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: MODEL)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CFG)

    calls = {}

    async def _native(*a, **kw):
        calls["native"] = True
        return "native"

    async def _aux(image_url, prompt, model=None, **kw):
        calls["aux"] = image_url
        return "aux"

    monkeypatch.setattr(vt, "_vision_analyze_native", _native)
    monkeypatch.setattr(vt, "vision_analyze_tool", _aux)
    monkeypatch.setattr(vt, "check_video_requirements", lambda: True)  # an aux backend resolves

    assert vt._should_use_native_vision_fast_path() is True
    registry.note_rejection(PROVIDER, MODEL)
    assert vt._should_use_native_vision_fast_path() is False

    out = await vt._handle_vision_analyze({"image_url": "/tmp/chart.png", "question": "q"})
    assert out == "aux" and calls == {"aux": "/tmp/chart.png"}


def test_aux_lane_sends_image_in_user_message():
    """The aux lane's request shape: the image rides a ``user`` message, which the relay
    accepts (only images inside ``tool`` messages are refused)."""
    from tools.vision_tools import _media_messages

    [msg] = _media_messages("describe", "image_url", "data:image/png;base64,AA==")
    assert msg["role"] == "user"
    assert any(p.get("type") == "image_url" for p in msg["content"])


# ─── agent-side gate (agent.vision_message_prep) ─────────────────────────────


def _prep_agent(**extra):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.provider, agent.model = PROVIDER, MODEL
    for k, v in extra.items():
        setattr(agent, k, v)
    return agent


def test_agent_gate_open_for_verified_route():
    assert _prep_agent()._provider_supports_vision_tool_messages() is True


def test_agent_gate_closed_by_registry():
    agent = _prep_agent()
    registry.note_rejection(PROVIDER, MODEL)
    assert agent._provider_supports_vision_tool_messages() is False


def test_agent_gate_closed_by_own_rejecting_set():
    agent = _prep_agent(_image_rejecting_models={(PROVIDER, MODEL)})
    assert agent._provider_supports_vision_tool_messages() is False


_AUX_DESCRIPTION = "A bar chart: revenue rises from 3 to 7 between Q1 and Q4."


@pytest.fixture
def aux_vision(monkeypatch):
    """Stub the aux vision LLM call (``tools.vision_tools.vision_analyze_tool``); records the sources."""
    import json

    seen: list = []

    async def _aux(image_url, user_prompt, model=None, **kw):
        seen.append(image_url)
        return json.dumps({"success": True, "analysis": _AUX_DESCRIPTION})

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _aux)
    return seen


def _vision_agent(monkeypatch, **extra):
    agent = _prep_agent(_anthropic_image_fallback_cache={}, **extra)
    monkeypatch.setattr(type(agent), "_model_supports_vision", lambda self: True)
    return agent


def _real_envelope():
    from tools.vision_tools import _build_native_vision_tool_result

    return _build_native_vision_tool_result("/tmp/x.png", "what trend?", "data:image/png;base64,AA==", 10)


def _assert_described(out):
    from agent.vision_claim_text import IMAGE_OMITTED_MARKER, has_native_vision_claim

    assert isinstance(out, str)
    assert _AUX_DESCRIPTION in out
    assert not has_native_vision_claim(out)
    assert "you can see it" not in out and "built-in vision" not in out
    assert IMAGE_OMITTED_MARKER not in out  # "no description is available" would contradict it


def test_closed_gate_swaps_image_for_aux_description(monkeypatch, aux_vision):
    agent = _vision_agent(monkeypatch)
    result = _real_envelope()
    assert isinstance(agent._tool_result_content_for_active_model("vision_analyze", result), list)
    registry.note_rejection(PROVIDER, MODEL)
    _assert_described(agent._tool_result_content_for_active_model("vision_analyze", result))
    assert len(aux_vision) == 1


def test_ttl_expiry_with_agent_set_still_yields_description(monkeypatch, aux_vision):
    """M1: the agent's own set never expires; the registry did. A native envelope produced in
    that window must still give the model the image's content, not a bare summary."""
    agent = _vision_agent(monkeypatch, _image_rejecting_models={(PROVIDER, MODEL)})
    assert registry.rejects_images(PROVIDER, MODEL) is False  # expired / never shared
    _assert_described(agent._tool_result_content_for_active_model("vision_analyze", _real_envelope()))


def test_send_path_strip_renews_registry_so_sides_cannot_diverge():
    """M1: every send for a route in the agent's rejecting set renews the shared record, so the
    tool side stays off the native lane exactly as long as this agent would strip the image."""
    from agent.message_sanitization import strip_images_for_rejecting_model
    from tools.vision_tools import _native_tool_result_images

    clock = _Clock()
    registry.note_rejection(PROVIDER, MODEL, clock=clock)
    clock.now += registry.DEFAULT_TTL_SECONDS + 1
    assert registry.rejects_images(PROVIDER, MODEL, clock=clock) is False

    agent = SimpleNamespace(provider=PROVIDER, model=MODEL, _image_rejecting_models={(PROVIDER, MODEL)})
    strip_images_for_rejecting_model(agent, [{"role": "user", "content": "hi"}])
    assert registry.rejects_images(PROVIDER, MODEL) is True
    assert _native_tool_result_images(PROVIDER, MODEL, _CFG) is False


def test_send_path_does_not_note_accepting_route():
    from agent.message_sanitization import strip_images_for_rejecting_model

    agent = SimpleNamespace(provider=PROVIDER, model=MODEL, _image_rejecting_models=set())
    assert strip_images_for_rejecting_model(agent, [{"role": "user", "content": "hi"}]) is False
    assert registry.rejects_images(PROVIDER, MODEL) is False


# ─── M2: list-type tool content rejection is shared too ─────────────────────


def _envelope_tool_msgs():
    import copy

    return [{"role": "tool", "tool_call_id": "c1", "content": copy.deepcopy(_real_envelope()["content"])}]


def test_list_content_rejection_closes_tool_side_native_lane():
    from tools.vision_tools import _native_tool_result_images

    assert _native_tool_result_images(PROVIDER, MODEL, _CFG) is True
    agent = _prep_agent()
    assert agent._try_strip_image_parts_from_tool_messages(_envelope_tool_msgs()) is True
    assert (PROVIDER, MODEL) in agent._no_list_tool_content_models
    assert registry.rejects_images(PROVIDER, MODEL) is True
    assert _native_tool_result_images(PROVIDER, MODEL, _CFG) is False
    # Same record on the agent side: a fresh agent (new session) reads it too.
    assert _prep_agent()._provider_supports_vision_tool_messages() is False


def test_413_strip_does_not_record_a_rejection():
    agent = _prep_agent()
    assert agent._try_strip_image_parts_from_tool_messages(_envelope_tool_msgs(), remember_model=False) is True
    assert registry.rejects_images(PROVIDER, MODEL) is False


def test_list_rejecting_agent_gets_description_not_bare_summary(monkeypatch, aux_vision):
    agent = _vision_agent(monkeypatch, _no_list_tool_content_models={(PROVIDER, MODEL)})
    _assert_described(agent._tool_result_content_for_active_model("vision_analyze", _real_envelope()))


# ─── M4: a recorded rejection never reshapes the tool list ──────────────────


@pytest.fixture
def main_route(monkeypatch):
    monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: PROVIDER)
    monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: MODEL)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CFG)


def test_vision_tools_stay_listed_after_rejection_without_aux(monkeypatch, main_route):
    import tools.browser_tool_install as bt_install
    import tools.vision_tools as vt

    monkeypatch.setattr(vt, "check_video_requirements", lambda: False)  # no aux backend
    monkeypatch.setattr(bt_install, "check_browser_requirements", lambda: True)
    before = (vt.check_vision_requirements(), bt_install.check_browser_vision_requirements())
    registry.note_rejection(PROVIDER, MODEL)
    assert vt._should_use_native_vision_fast_path() is False  # the lane IS closed…
    after = (vt.check_vision_requirements(), bt_install.check_browser_vision_requirements())
    assert before == after == (True, True)  # …but the tool list does not move


def test_registry_listing_stable_across_rejection(monkeypatch, main_route):
    """Through the real registry: ``vision_analyze``'s schema is emitted before and after."""
    import tools.vision_tools as vt
    from tools.registry import invalidate_check_fn_cache, registry as tool_registry

    monkeypatch.setattr(vt, "check_video_requirements", lambda: False)

    def _names():
        invalidate_check_fn_cache()  # bypass the ~30 s probe cache: re-evaluate check_fn each time
        return [d["function"]["name"] for d in tool_registry.get_definitions({"vision_analyze"}, quiet=True)]

    assert _names() == ["vision_analyze"]
    registry.note_rejection(PROVIDER, MODEL)
    assert _names() == ["vision_analyze"]


def test_profile_veto_still_hides_native_half():
    """Fail-closed stays: the static profile veto (exact-id exceptions only) is honoured even
    when transient rejections are ignored."""
    from tools.vision_tools import _accepts_tool_result_images

    for model in ("glm-5.3-flash", "DeepSeek-V4.1-Flash", "deepseek-v4.1-flash-preview"):
        assert _accepts_tool_result_images(PROVIDER, model, _CFG, honor_rejections=False) is False
    assert _accepts_tool_result_images(PROVIDER, MODEL, _CFG, honor_rejections=False) is True


@pytest.mark.asyncio
async def test_suspended_lane_without_aux_returns_explicit_error(monkeypatch, main_route):
    import json

    import tools.vision_tools as vt

    monkeypatch.setattr(vt, "check_video_requirements", lambda: False)
    registry.note_rejection(PROVIDER, MODEL)

    async def _never(*a, **kw):
        raise AssertionError("no lane should run")

    monkeypatch.setattr(vt, "_vision_analyze_native", _never)
    monkeypatch.setattr(vt, "vision_analyze_tool", _never)
    out = json.loads(await vt._handle_vision_analyze({"image_url": "/tmp/chart.png", "question": "q"}))
    assert out["success"] is False
    assert "temporarily unavailable" in out["error"] and "not visible to you" in out["error"]


# ─── turn_recovery records into the registry ─────────────────────────────────


def test_turn_recovery_notes_rejection():
    from agent.turn_recovery import recover_before_classification

    class _Err(Exception):
        status_code = 400
        body = {"error": {"message": "This model does not support images"}}

    agent = SimpleNamespace(
        provider=PROVIDER, model=MODEL, _image_rejecting_models=set(), log_prefix="",
        _unicode_sanitization_passes=0, api_mode="chat_completions", quiet_mode=True,
    )
    agent._vprint = lambda *a, **kw: None
    agent._emit_status = lambda *a, **kw: None
    retry, _ = recover_before_classification(
        agent, _Err("does not support images"), messages=[], api_messages=[],
        api_kwargs={}, active_system_prompt="",
    )
    assert retry is True
    assert (PROVIDER, MODEL) in agent._image_rejecting_models
    assert registry.rejects_images(PROVIDER, MODEL) is True
