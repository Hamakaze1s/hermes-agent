"""Tests for reactive multimodal-tool-content recovery.

Covers the full chain for providers that reject list-type content in
``role: "tool"`` messages (Xiaomi MiMo's 400 "text is not set", etc.):

  1. agent/error_classifier.py: 400 with the right wording classifies as
     ``FailoverReason.multimodal_tool_content_unsupported``.
  2. run_agent._try_strip_image_parts_from_tool_messages downgrades tool
     messages whose ``content`` is a list-with-image to a string text
     summary, in-place, and records the active (provider, model) in
     ``self._no_list_tool_content_models`` so future tool results in this
     session preemptively downgrade.
  3. run_agent._tool_result_content_for_active_model short-circuits to a
     text summary when the (provider, model) is in the cache, even though
     ``_model_supports_vision`` returns True — avoiding a wasted round
     trip on every subsequent screenshot in the session.

The end-to-end retry loop wiring (`conversation_loop.py`) is exercised by
the classifier signal + helper-mutation tests; the integration only adds
a trivial flag-and-continue around the existing pattern used for
``image_too_large`` recovery.

See: https://github.com/NousResearch/hermes-agent/issues/27344
"""

from __future__ import annotations


from agent.error_classifier import FailoverReason, classify_api_error


class _FakeApiError(Exception):
    """Stand-in for an openai.BadRequestError with status_code + body."""

    def __init__(self, status_code: int, message: str, body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {"error": {"message": message}}
        self.response = None


def _make_agent(provider: str = "xiaomi", model: str = "mimo-v2.5"):
    """Build a bare AIAgent for method-level testing, no provider setup."""
    from run_agent import AIAgent
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    return agent


# ─── Strip helper ────────────────────────────────────────────────────────────


class TestStripImagePartsHelper:


    def test_tool_message_with_string_content_unchanged(self):
        agent = _make_agent()
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": "plain string result"},
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is False
        assert msgs[0]["content"] == "plain string result"

    def test_tool_message_list_without_image_unchanged(self):
        """List content with only text parts is left alone — caller surfaces
        the original error if this turns out to also be rejected."""
        agent = _make_agent()
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "hello"},
            ]},
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is False



    def test_records_provider_model_in_session_cache(self):
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5")
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "summary"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
            ]},
        ]
        agent._try_strip_image_parts_from_tool_messages(msgs)
        assert ("xiaomi", "mimo-v2.5") in agent._no_list_tool_content_models

    def test_native_envelope_strip_does_not_claim_the_model_can_see(self):
        """A downgraded native envelope must never tell the model it can see the image.

        The native envelope's text part is "Image loaded into your context — you can see it
        natively now…"; keeping that sentence after the bytes are stripped makes the model
        describe a picture it never received (silent blindness), which is worse than an
        honest error.
        """
        agent = _make_agent(provider="opencode-go", model="deepseek-v4.1-flash")
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "Image loaded into your context — you can see it natively "
                                         "now. Use your built-in vision to answer the user."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
            ]},
        ]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is True
        text = msgs[0]["content"]
        assert isinstance(text, str)
        # No surviving sentence may imply the image arrived…
        assert "you can see it natively now" not in text
        assert "loaded into your context" not in text
        assert "built-in vision" not in text
        # …and removing a sentence must not leave a stray dash/period shell behind.
        assert "— ." not in text
        assert not text.rstrip().endswith("—")
        assert "not visible to you" in text.lower()
        assert "image_url" not in text

    def test_native_envelope_keeps_question_and_note_intact(self):
        """Only the claim sentences go: the trailing Question/Note blocks are content."""
        agent = _make_agent(provider="opencode-go", model="deepseek-v4.1-flash")
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": (
                    "Image loaded into your context — you can see it natively now. "
                    "Use your built-in vision to answer the user.\n\n"
                    "Question: 这个表格里 B 的延迟是多少？\n\n"
                    "Note: downscaled from 4000px."
                )},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
            ]},
        ]
        agent._try_strip_image_parts_from_tool_messages(msgs)
        text = msgs[0]["content"]
        assert "Question: 这个表格里 B 的延迟是多少？" in text
        assert "downscaled from 4000px." in text  # trailing period survives
        assert "\n\nQuestion:" in text  # block structure survives
        assert "you can see it" not in text

    def test_non_native_tool_text_is_left_verbatim(self):
        agent = _make_agent()
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "screenshot ready"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
            ]},
        ]
        agent._try_strip_image_parts_from_tool_messages(msgs)
        assert msgs[0]["content"] == "screenshot ready"

    def test_only_tool_messages_get_downgraded(self):
        """User / assistant messages with list-type content are out of
        scope — they're handled by the existing image-routing path."""
        agent = _make_agent()
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
            ]},
            {"role": "tool", "tool_call_id": "x", "content": [
                {"type": "text", "text": "summary"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,Y"}},
            ]},
        ]
        agent._try_strip_image_parts_from_tool_messages(msgs)
        # User message untouched.
        assert isinstance(msgs[0]["content"], list)
        assert any(p.get("type") == "image_url" for p in msgs[0]["content"])
        # Tool message downgraded.
        assert isinstance(msgs[1]["content"], str)
        assert "summary" in msgs[1]["content"]



# ─── Short-circuit on cached models ──────────────────────────────────────────


class TestToolResultContentShortCircuit:
    """Once the session has learned that (provider, model) rejects list
    content, ``_tool_result_content_for_active_model`` returns a text
    summary even though ``_model_supports_vision`` reports True.
    """

    def _multimodal_result(self, png_b64: str = "iVBORw0KGgoAAAA"):
        return {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "capture mode=som 800x600 app=Safari"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
            ],
            "text_summary": "capture mode=som 800x600 app=Safari",
            "meta": {"mode": "som", "width": 800, "height": 600, "elements": 5,
                     "png_bytes": 1024},
        }

    def test_returns_text_summary_for_xiaomi_proactively(self, monkeypatch):
        """Xiaomi MiMo rejects list-type tool content, so even with an
        empty cache, _tool_result_content_for_active_model should
        proactively downgrade to a text summary."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5")
        agent._no_list_tool_content_models = set()  # explicit empty
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        out = agent._tool_result_content_for_active_model(
            "computer_use", self._multimodal_result()
        )
        # Proactive downgrade: text summary instead of list with images.
        assert isinstance(out, str)
        assert "data:image" not in out
        assert "image_url" not in out





# ─── Classifier ──────────────────────────────────────────────────────────────


class TestRecoveryEndToEndClassification:
    """Lock in that the patterns used by the recovery path classify to
    the right ``FailoverReason``. (The recovery hook in
    ``agent.conversation_loop`` consumes this reason directly.)
    """

    def test_xiaomi_mimo_classifies(self):
        err = _FakeApiError(
            status_code=400,
            message=(
                "Error code: 400 - {'error': {'code': '400', 'message': "
                "'Param Incorrect', 'param': 'text is not set', 'type': ''}}"
            ),
        )
        result = classify_api_error(err, provider="xiaomi", model="mimo-v2.5")
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported
        assert result.retryable is True

    def test_alibaba_variant_classifies(self):
        err = _FakeApiError(
            status_code=400,
            message="tool_call.content must be string",
        )
        result = classify_api_error(err, provider="alibaba", model="qwen3.5-plus")
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported

    def test_opencode_go_422_console_go_classifies(self):
        """Regression test for #104731: Console Go relay HTTP 422 on list-type tool content."""
        err = _FakeApiError(
            status_code=422,
            message=(
                "Error code: 422 - {'error': {'param': 'messages.67.tool.content.str', "
                "'type': 'invalid_request_error', 'message': 'Error from provider (Console Go): "
                "Upstream request failed: [invalid_request_error] Input should be a valid string'}}"
            ),
        )
        result = classify_api_error(err, provider="opencode-go", model="deepseek-v4-flash-vision-exp")
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported
        assert result.retryable is True

    def test_nvidia_nim_serde_tool_content_classifies(self):
        """Regression test for #111231 (bug 2): NVIDIA NIM's Rust gateway rejects
        list-type tool content without naming the field — the serde text names the
        *enum* instead. Verbatim provider wording from the report."""
        err = _FakeApiError(
            status_code=400,
            message=(
                "HTTP 400: Failed to deserialize the JSON body into the target type: "
                "data did not match any variant of untagged enum "
                "ChatCompletionRequestToolMessageContent at line 1 column 1974809"
            ),
        )
        result = classify_api_error(err, provider="nvidia", model="moonshotai/kimi-k3")
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported
        assert result.retryable is True

    def test_nvidia_nim_serde_other_enum_not_misclassified(self):
        """A NIM serde rejection naming a different enum must stay a format error —
        only the tool-message-content enum means list-type tool content."""
        err = _FakeApiError(
            status_code=400,
            message=(
                "HTTP 400: Failed to deserialize the JSON body into the target type: "
                "data did not match any variant of untagged enum ChatCompletionRequestMessage"
            ),
        )
        result = classify_api_error(err, provider="nvidia", model="moonshotai/kimi-k3")
        assert result.reason != FailoverReason.multimodal_tool_content_unsupported

class TestOpenCodeGoProactiveToolResultDowngrade:
    def _multimodal_result(self, png_b64: str = "iVBORw0KGgoAAAA"):
        return {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "detected 2 objects"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
            ],
            "text_summary": "detected 2 objects",
            "meta": {"png_bytes": 1024},
        }

    def test_returns_text_summary_for_opencode_go_proactively(self, monkeypatch):
        """OpenCode Go rejects list-type tool content, so _tool_result_content_for_active_model
        should proactively downgrade to a text summary (#104731)."""
        agent = _make_agent(provider="opencode-go", model="deepseek-v4-flash-vision-exp")
        agent._no_list_tool_content_models = set()
        monkeypatch.setattr(agent, "_model_supports_vision", lambda: True)
        out = agent._tool_result_content_for_active_model("vision_analyze", self._multimodal_result())
        assert isinstance(out, str)
        assert "data:image" not in out
        assert "image_url" not in out
