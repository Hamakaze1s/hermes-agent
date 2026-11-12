"""opencode-go: a provider-wide tool-image veto with an exact-id, live-verified lift.

The relay fronts several backends. MiMo/GLM reject list-type tool content ("text is
not set"), while the DeepSeek backend accepts image parts inside a tool message —
verified 2026-11-12 with 20/20 successful reads across 20 distinct
``x-opencode-session`` values on the direct relay. Only the exact verified id lifts the
veto; everything else (other backends, unknown ids, empty model, aggregator/alias
prefixes) stays fail-closed because a rejected row poisons the whole history (#89981).
"""
from __future__ import annotations

import pytest

from providers import routed_model_rejects_vision_tool_messages


@pytest.fixture(autouse=True)
def _load_profiles():
    import providers

    providers.list_providers()


def test_verified_deepseek_lane_accepts_tool_images():
    assert routed_model_rejects_vision_tool_messages("opencode-go", "deepseek-v4.1-flash") is False


def test_other_backends_stay_vetoed():
    for model in ("mimo-v2.5", "glm-5.3-flash", "qwen3.8-max"):
        assert routed_model_rejects_vision_tool_messages("opencode-go", model) is True


def test_unknown_model_stays_fail_closed():
    assert routed_model_rejects_vision_tool_messages("opencode-go", "some-future-model") is True


def test_empty_or_missing_model_stays_fail_closed():
    for model in ("", "   ", None):
        assert routed_model_rejects_vision_tool_messages("opencode-go", model) is True


def test_id_variants_are_not_matched():
    """Exact ids only: the evidence covers one spelling, and a wrong guess here lets a
    backend that rejects the shape poison the history."""
    for model in ("DeepSeek-V4.1-Flash", "deepseek-v4.1-flash-2026", "opencode-go/deepseek-v4.1-flash"):
        assert routed_model_rejects_vision_tool_messages("opencode-go", model) is True


def test_aggregator_and_prefix_forms_stay_vetoed():
    """The exception is verified for the direct relay only; a routing aggregator or an
    alias prefix is an untested surface, so the veto must hold there."""
    assert routed_model_rejects_vision_tool_messages(
        "openrouter", "opencode-go/deepseek-v4.1-flash") is True


def test_lift_does_not_leak_across_providers():
    assert routed_model_rejects_vision_tool_messages("xiaomi", "deepseek-v4.1-flash") is True
