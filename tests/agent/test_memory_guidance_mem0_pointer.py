"""Local patch P1a: the model must be told where architecture-2.0 knowledge goes.

Upstream rebuilds the memory guidance with ``build_memory_guidance()`` and the
runtime injects THAT (``system_prompt._tool_guidance_block``), not the legacy
``MEMORY_GUIDANCE`` alias. The mem0 default-write pointer therefore has to ride
the builder's output — attaching it to the alias only would compile, pass a
smoke import, and still never reach a single prompt. These tests pin the
runtime path, not the alias.
"""

from __future__ import annotations

from agent.prompt_builder import MEMORY_GUIDANCE, build_memory_guidance


def test_builder_output_carries_the_mem0_pointer():
    text = build_memory_guidance(True, True)
    assert "mem0" in text
    assert "default write target" in text
    assert "type metadata" in text


def test_legacy_alias_shares_the_builder_output():
    assert "mem0" in MEMORY_GUIDANCE


def test_pointer_is_absent_when_the_builtin_store_is_disabled():
    # memory_enabled=False is the user-profile-only shape: the built-in store is
    # gone, so pointing at mem0 as "the default write target" would be wrong.
    assert "mem0" not in build_memory_guidance(False, True)


def test_runtime_guidance_block_carries_the_pointer():
    """The regression this file exists for: the runtime injection point."""
    from agent import system_prompt

    class _Agent:
        valid_tool_names = {"memory", "session_search", "skill_manage"}
        model = "test-model"
        provider = "test-provider"
        base_url = ""
        api_key = ""

    block = system_prompt._tool_guidance_block(_Agent())
    assert block, "tool guidance block should not be empty for these tools"
    assert "mem0" in block, (
        "the mem0 pointer must reach the runtime guidance block — attaching it "
        "to the MEMORY_GUIDANCE alias instead of build_memory_guidance() makes "
        "it dead code"
    )
