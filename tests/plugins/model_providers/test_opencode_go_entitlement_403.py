"""An OpenCode Go account without a subscription must bench its key and rotate, never retry.

The relay answers two unrelated conditions with one envelope: HTTP 403 whose ``error.type``
(and usually ``code``) is ``server_error`` and whose message is wrapped in
"Upstream request failed: ". One is the relay's own upstream returning a body it could not
parse (a blip), the other is "An active OpenCode Go subscription is required to use Go models."
(the account cannot serve Go traffic). Only the message text separates them, so the profile
hook claims the entitlement body and leaves the blip to core's built-in verdict.

Regression context: NousResearch/hermes-agent#117869 (and the open PRs that would classify
every ``code=server_error`` 403 as transient, which would also swallow this body).
"""

from __future__ import annotations

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error

BASE = "https://opencode.ai/zen/go/v1"

ENTITLEMENT_403 = {
    "error": {
        "type": "server_error",
        "message": "Upstream request failed: An active OpenCode Go subscription is required to use Go models.",
    }
}
BLIP_403 = {
    "error": {
        "type": "server_error",
        "code": "server_error",
        "message": "Upstream request failed: [server_error] Upstream response was not valid JSON",
    }
}
ZEN_FREE_TIER_403 = {
    "type": "error",
    "error": {
        "type": "FreeTierError",
        "message": "OpenCode's free tier can only be used from within OpenCode.",
    },
}
EXPIRED_KEY_401 = {
    "error": {
        "type": "server_error",
        "message": "This API key has expired. Create a new API key in OpenCode Console.",
    }
}


def _error(status: int, body: dict) -> openai.APIStatusError:
    resp = httpx.Response(status, json=body, request=httpx.Request("POST", BASE + "/chat/completions"))
    return openai.APIStatusError("relay refused", response=resp, body=body)


def _classify(status: int, body: dict, *, provider: str = "opencode-go"):
    return classify_api_error(_error(status, body), provider=provider, model="deepseek-v4.1-flash", base_url=BASE)


def test_no_subscription_403_benches_the_key_and_rotates():
    c = _classify(403, ENTITLEMENT_403)
    assert (c.reason, c.retryable) == (FailoverReason.billing, False)
    assert c.should_rotate_credential is True and c.should_fallback is True


def test_no_subscription_403_matches_regardless_of_case():
    body = {"error": {"type": "server_error",
                      "message": "Upstream request failed: AN ACTIVE OPENCODE GO SUBSCRIPTION IS REQUIRED TO USE GO MODELS."}}
    assert _classify(403, body).reason == FailoverReason.billing


def test_relay_blip_403_keeps_its_builtin_verdict():
    """The other half of the shared envelope: the credential is healthy, so the hook declines."""
    c = _classify(403, BLIP_403)
    assert c.reason == FailoverReason.auth
    assert c.should_rotate_credential is False


def test_expired_key_401_is_untouched():
    c = _classify(401, EXPIRED_KEY_401)
    assert (c.reason, c.should_rotate_credential) == (FailoverReason.auth, True)


@pytest.mark.parametrize("provider", ["opencode-zen", "openrouter"])
def test_the_hook_claims_only_the_opencode_go_route(provider):
    """The Zen free-tier refusal is a different shape, and no other route gets this branch."""
    assert _classify(403, ZEN_FREE_TIER_403, provider=provider).reason == FailoverReason.auth
    assert _classify(403, ENTITLEMENT_403, provider=provider).reason == FailoverReason.auth
