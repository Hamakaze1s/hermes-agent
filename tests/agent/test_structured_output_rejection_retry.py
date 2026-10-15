"""Regression tests for the structured-output rejection retry in
``agent.auxiliary_client``.

Auxiliary callers (title generation, plugin structured completions) send an
OpenAI ``response_format`` request field. Some providers reject the field, or
its Anthropic translation, with a hard 400:

  * vLLM gateways translate ``response_format: json_schema`` into
    ``guided_grammar`` and fail when the grammar backend is absent
    (``compile_grammar_error: No module named 'xgrammar'``, #82816).
  * Some OpenAI-compatible endpoints answer
    ``This response_format type is unavailable now`` (#82816).
  * Anthropic-compatible gateways that predate structured outputs reject the
    translated ``output_config`` field with
    ``output_config: Extra inputs are not permitted`` (the documented case is
    the ``bedrock-mantle`` Messages endpoint).
  * OpenAI-compatible gateways that validate the body with a strict pydantic
    model reject the OBJECT-form ``response_format.json_schema`` by shape --
    ``422 ... body.response_format.json_schema: str type expected`` -- instead
    of naming the feature, so the error never mentions an unsupported option.
  * OpenCode's relay answers from some of its upstream endpoints with an
    opaque ``{"model": "<id>"}`` body — no phrasing to match, so the shape is
    recognized instead (``_is_model_echo_error``). Same field, same reaction.

Callers tolerate an unconstrained reply: the title prompt demands bare JSON
and ``_extract_title_text`` has a loose-JSON fallback. The fix is reactive,
like the temperature retry: when the provider rejects the structured-output
field, retry once without it. These tests lock in that behaviour for both
sync and async paths.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    _is_model_echo_error,
    _is_structured_output_rejection,
    _without_structured_output_format,
)


_TITLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "session_title",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}


class _ApiStatusError(Exception):
    """Stand-in for ``openai.APIStatusError`` — status code plus parsed body."""

    def __init__(self, body, status_code=400):
        self.body = body
        self.status_code = status_code
        super().__init__(f"Error code: {status_code} - {body}")


class TestIsStructuredOutputRejection:
    """The detector must match the phrasings providers actually return."""

    @pytest.mark.parametrize("message", [
        # vLLM guided_grammar / xgrammar (#82816, verbatim from the report)
        (
            "Error code: 400 - {'error': {'message': 'guided_grammar "
            '\'{"additionalProperties":false}\' has compile_grammar_error: '
            "No module named 'xgrammar'', 'type': 'invalid_request_error'}}"
        ),
        # Second endpoint from the same report
        "HTTP 400: This response_format type is unavailable now",
        # Strict Anthropic-wire gateways rejecting the raw OpenAI field
        "HTTP 400: response_format: Extra inputs are not permitted",
        # Gateways that predate output_config (bedrock-mantle documented case)
        "HTTP 400: output_config: Extra inputs are not permitted",
        # Generic unsupported-parameter phrasings for both field names
        "Unsupported parameter: response_format",
        "output_config is not supported",
        # Strict pydantic gateway rejecting the object-form json_schema by SHAPE
        # (422, verbatim body from the gateway) -- no unsupported-parameter wording
        'HTTP 422: {"detail":[{"loc":["body","response_format","json_schema"],'
        '"msg":"str type expected","type":"type_error.str"}]}',
        # Gemini native generationConfig wording (verbatim 400s from generativelanguage)
        "Gemini HTTP 400 (INVALID_ARGUMENT): Function calling with a response mime type: 'application/json' is unsupported",
        "Gemini HTTP 400 (INVALID_ARGUMENT): Invalid JSON payload received. Unknown name \"response_json_schema\" at 'generation_config'",
        "Gemini HTTP 400 (INVALID_ARGUMENT): Invalid value at 'generation_config.response_schema.properties[0].value.type'",
    ])
    def test_matches_real_provider_messages(self, message):
        assert _is_structured_output_rejection(RuntimeError(message)) is True

    @pytest.mark.parametrize("message", [
        # Unrelated 400s must NOT trigger a silent schema downgrade
        "HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant'",
        "HTTP 400: Unsupported parameter: temperature",
        "max_tokens is too large for this model",
        "Rate limit exceeded",
        "Connection reset by peer",
        # Alternation errors that happen to mention messages
        "messages: Extra inputs are not permitted",
    ])
    def test_does_not_match_unrelated_errors(self, message):
        assert _is_structured_output_rejection(RuntimeError(message)) is False

    def test_does_not_match_non_400_statuses(self):
        exc = RuntimeError("output_config: Extra inputs are not permitted")
        exc.status_code = 500
        assert _is_structured_output_rejection(exc) is False


class TestModelEchoRejection:
    """The OpenCode relay's opaque 400: body is the echoed model id, nothing else.

    Live shape (2026-10-12, verified by replaying an identical title request with
    and without ``response_format`` against one session-affinity id):

        {"model": "deepseek-v4.1-flash"}

    No phrasing to match, so the detector keys on the shape. The retry stays safe
    because callers only strip the field when the request actually carried it.
    """

    def test_matches_the_live_opencode_body(self):
        exc = _ApiStatusError({"model": "deepseek-v4.1-flash"})
        assert _is_model_echo_error(exc) is True
        assert _is_structured_output_rejection(exc) is True

    def test_matches_the_same_body_carried_as_text(self):
        exc = _ApiStatusError('{"model": "deepseek-v4.1-flash"}')
        assert _is_structured_output_rejection(exc) is True

    def test_requires_an_error_status(self):
        assert _is_model_echo_error(_ApiStatusError({"model": "m"}, status_code=200)) is False
        assert _is_model_echo_error(_ApiStatusError({"model": "m"}, status_code=429)) is False
        # No status at all is not enough evidence to drop the schema contract.
        assert _is_model_echo_error(RuntimeError("{'model': 'm'}")) is False

    def test_requires_the_body_to_be_a_non_empty_mapping(self):
        assert _is_model_echo_error(_ApiStatusError({})) is False
        assert _is_model_echo_error(_ApiStatusError({"error": {"message": "boom"}})) is False
        assert _is_model_echo_error(_ApiStatusError("not json")) is False

    @pytest.mark.parametrize("body", [
        {"model": "m", "error": {"message": "boom"}},
        {"model": "m", "message": "boom"},
        {"model": "m", "detail": "invalid request"},
        {"model": "m", "type": "invalid_request_error"},
        {"model": "m", "code": "invalid_request_error"},
        {"model": "m", "param": "response_format"},
    ])
    def test_bodies_that_explain_themselves_are_not_opaque(self, body):
        assert _is_model_echo_error(_ApiStatusError(body)) is False

    def test_unrelated_400_without_model_echo_is_untouched(self):
        exc = _ApiStatusError({"error": {"message": "Invalid value: 'tool'"}})
        assert _is_structured_output_rejection(exc) is False


class TestWithoutStructuredOutputFormat:
    """The kwargs scrubber removes the field on both call shapes."""

    def test_removes_extra_body_entry_and_keeps_siblings(self):
        kwargs = {
            "model": "m",
            "extra_body": {
                "response_format": dict(_TITLE_RESPONSE_FORMAT),
                "metadata": {"user_id": "u1"},
            },
        }
        result = _without_structured_output_format(kwargs)
        assert result is not None
        assert result["extra_body"] == {"metadata": {"user_id": "u1"}}
        # The input dict is not mutated.
        assert "response_format" in kwargs["extra_body"]

    def test_drops_extra_body_entirely_when_it_becomes_empty(self):
        kwargs = {
            "model": "m",
            "extra_body": {"response_format": dict(_TITLE_RESPONSE_FORMAT)},
        }
        result = _without_structured_output_format(kwargs)
        assert result is not None
        assert "extra_body" not in result

    def test_removes_top_level_kwarg(self):
        kwargs = {"model": "m", "response_format": dict(_TITLE_RESPONSE_FORMAT)}
        result = _without_structured_output_format(kwargs)
        assert result is not None
        assert "response_format" not in result

    def test_returns_none_when_nothing_to_remove(self):
        assert _without_structured_output_format({"model": "m"}) is None
        assert _without_structured_output_format(
            {"model": "m", "extra_body": {"metadata": {}}}
        ) is None


def _dummy_response():
    return {"ok": True}


class TestCallLlmStructuredOutputRetry:
    """``call_llm`` retries once without the field and returns on success."""

    def _setup(self, first_exc):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create.side_effect = [
            first_exc, _dummy_response(),
        ]
        return client

    @pytest.mark.parametrize("error_message", [
        # vLLM guided_grammar (#82816)
        "Error code: 400 - guided_grammar has compile_grammar_error: "
        "No module named 'xgrammar'",
        # Second endpoint flavor from the same report
        "HTTP 400: This response_format type is unavailable now",
        # Strict gateway that rejects the translated Anthropic field
        "HTTP 400: output_config: Extra inputs are not permitted",
    ])
    def test_retries_once_without_response_format(self, error_message):
        client = self._setup(RuntimeError(error_message))

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
        ):
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"response_format": dict(_TITLE_RESPONSE_FORMAT)},
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        first_eb = first_kwargs.get("extra_body") or {}
        retry_eb = retry_kwargs.get("extra_body") or {}
        assert "response_format" in first_eb
        assert "response_format" not in retry_eb
        assert "response_format" not in retry_kwargs
        assert retry_kwargs["model"] == first_kwargs["model"]

    def test_retries_once_on_the_opencode_model_echo_400(self):
        """The live 2026-10-12 OpenCode shape retries instead of hard-failing.

        Body is the echoed model id and nothing else, so the retry decision is
        shape-based; the request's own ``response_format`` is what authorizes it.
        """
        client = MagicMock()
        client.base_url = "https://opencode.ai/zen/go/v1"
        client.chat.completions.create.side_effect = [
            _ApiStatusError({"model": "deepseek-v4.1-flash"}),
            _dummy_response(),
        ]

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
        ):
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"response_format": dict(_TITLE_RESPONSE_FORMAT)},
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert "response_format" in (first_kwargs.get("extra_body") or {})
        assert "response_format" not in (retry_kwargs.get("extra_body") or {})
        assert "response_format" not in retry_kwargs

    def test_opaque_400_does_not_retry_when_no_field_was_sent(self):
        """Shape alone must not authorize a retry the scrubber can't change."""
        client = MagicMock()
        client.base_url = "https://opencode.ai/zen/go/v1"
        client.chat.completions.create.side_effect = _ApiStatusError(
            {"model": "deepseek-v4.1-flash"}
        )

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(Exception, match="deepseek-v4.1-flash"):
                call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=64,
                )
        assert client.chat.completions.create.call_count == 1

    def test_unrelated_400_does_not_strip_response_format(self):
        """Unrelated 400s must not silently downgrade the schema contract."""
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create.side_effect = RuntimeError(
            "HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant'"
        )

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(RuntimeError, match="Invalid value"):
                call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=64,
                    extra_body={
                        "response_format": dict(_TITLE_RESPONSE_FORMAT),
                    },
                )
        assert client.chat.completions.create.call_count == 1

    def test_no_retry_when_no_response_format_was_sent(self):
        """A rejection with no field in the request must not loop a retry."""
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create.side_effect = RuntimeError(
            "HTTP 400: output_config: Extra inputs are not permitted"
        )

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(RuntimeError):
                call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=64,
                )
        assert client.chat.completions.create.call_count == 1


class TestAsyncCallLlmStructuredOutputRetry:
    """``async_call_llm`` mirror of the sync retry semantics."""

    @pytest.mark.asyncio
    async def test_async_retries_once_without_response_format(self):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create = AsyncMock(side_effect=[
            RuntimeError(
                "Error code: 400 - guided_grammar has compile_grammar_error: "
                "No module named 'xgrammar'"
            ),
            _dummy_response(),
        ])

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
        ):
            result = await async_call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"response_format": dict(_TITLE_RESPONSE_FORMAT)},
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.await_count == 2
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert "response_format" in (first_kwargs.get("extra_body") or {})
        assert "response_format" not in (retry_kwargs.get("extra_body") or {})
        assert "response_format" not in retry_kwargs

    @pytest.mark.asyncio
    async def test_async_retries_once_on_the_opencode_model_echo_400(self):
        client = MagicMock()
        client.base_url = "https://opencode.ai/zen/go/v1"
        client.chat.completions.create = AsyncMock(side_effect=[
            _ApiStatusError({"model": "deepseek-v4.1-flash"}),
            _dummy_response(),
        ])

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
        ):
            result = await async_call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"response_format": dict(_TITLE_RESPONSE_FORMAT)},
            )

        assert result == {"ok": True}
        assert client.chat.completions.create.await_count == 2
        retry_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert "response_format" not in (retry_kwargs.get("extra_body") or {})
        assert "response_format" not in retry_kwargs

    @pytest.mark.asyncio
    async def test_async_unrelated_400_does_not_retry(self):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("HTTP 400: Invalid value: 'tool'"),
        )

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model",
                  return_value=("openai-codex", "gpt-5.5", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client",
                  return_value=(client, "gpt-5.5")),
            patch("agent.auxiliary_client._validate_llm_response",
                  side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._try_payment_fallback",
                  return_value=None),
        ):
            with pytest.raises(RuntimeError, match="Invalid value"):
                await async_call_llm(
                    task="title_generation",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=64,
                    extra_body={
                        "response_format": dict(_TITLE_RESPONSE_FORMAT),
                    },
                )
        assert client.chat.completions.create.await_count == 1
