"""Tests for check_for_tool_uses.

HyperAgents' tool protocol asks the model for a ``<json>{tool_name,tool_input}</json>``
block, but Claude 4.5-class models (e.g. Haiku 4.5) instead emit their native
``<function_calls>``/``<invoke>`` tool XML. The strict ``<json>...</json>`` parser
dropped every such call, so no tool ever executed and every arc/polyglot/swebench
task scored 0 with a missing prediction. These pin that both formats parse.
"""

from agent.llm_withtools import check_for_tool_uses


def test_legacy_json_block():
    r = 'ok\n<json>\n{"tool_name": "bash", "tool_input": {"command": "ls"}}\n</json>'
    assert check_for_tool_uses(r) == [{"tool_name": "bash", "tool_input": {"command": "ls"}}]


def test_variant_b_json_closed_by_function_calls():
    # Haiku hybrid: opens <json> but closes with </function_calls> (no </json>).
    r = (
        "<function_calls>\n<json>\n"
        '{"tool_name": "editor", "tool_input": {"command": "view", "path": "/testbed"}}\n'
        "</function_calls>"
    )
    assert check_for_tool_uses(r) == [
        {"tool_name": "editor", "tool_input": {"command": "view", "path": "/testbed"}}
    ]


def test_variant_a_native_invoke():
    # Pure Anthropic-native tool XML, no <json> at all.
    r = (
        "<function_calls>\n<invoke name=\"bash\">\n"
        "<parameter name=\"command\">cd /testbed && ls</parameter>\n"
        "</invoke>\n</function_calls>"
    )
    assert check_for_tool_uses(r) == [
        {"tool_name": "bash", "tool_input": {"command": "cd /testbed && ls"}}
    ]


def test_native_invoke_multiple_params():
    r = (
        "<invoke name=\"editor\">"
        "<parameter name=\"command\">create</parameter>"
        "<parameter name=\"path\">/testbed/prediction.json</parameter>"
        "</invoke>"
    )
    assert check_for_tool_uses(r) == [
        {"tool_name": "editor", "tool_input": {"command": "create", "path": "/testbed/prediction.json"}}
    ]


def test_first_tool_call_ordering_preserved():
    # The loop uses tool_uses[:1] by default, so the first emitted must be first.
    r = (
        "<invoke name=\"bash\"><parameter name=\"command\">first</parameter></invoke>"
        '<json>{"tool_name": "editor", "tool_input": {"command": "second"}}</json>'
    )
    assert check_for_tool_uses(r)[0] == {"tool_name": "bash", "tool_input": {"command": "first"}}


def test_nested_braces_in_tool_input():
    r = '<json>{"tool_name": "bash", "tool_input": {"command": "echo {nested}"}}</json>'
    assert check_for_tool_uses(r) == [
        {"tool_name": "bash", "tool_input": {"command": "echo {nested}"}}
    ]


def test_no_tool_calls_returns_none():
    assert check_for_tool_uses("just some prose, no tools here") is None
