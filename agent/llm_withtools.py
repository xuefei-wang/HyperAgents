"""Tool-using chat loop on top of litellm.completion(tools=[...]).

Replaces the upstream prose-regex protocol (where the model emits
<json>{"tool_name": ..., "tool_input": ...}</json> in its text output and
the loader regex-parses it) with litellm's first-class tool-calling API:
``tools=[{"type": "function", ...}]`` plus ``tool_choice="auto"``. Anthropic,
OpenAI, Gemini, and any other provider litellm supports go through the same
code path; the model gets schema-validated tool calls and we get structured
``tool_calls`` back.

Why we needed this: Claude Haiku 4.5 doesn't reliably comply with the prose
protocol for short tool prompts -- it emits ``{"tool_name": "bash", "command":
"..."}`` with params at the top level instead of nested under ``"tool_input"``,
which the upstream regex parser silently drops. Reproduced 20/20 on a simple
``ls`` prompt against pristine HA. Once the parser drops the tool call, the
model hallucinates an answer with no tool ever firing -> empty patches all
the way down -> the meta-agent cannot self-improve.

Returns ``new_msg_history`` as ``[{role, text}, ...]`` with the final assistant
message wrapped as ``<json>{"response": "<text>"}</json>`` so existing
``extract_jsons`` consumers (e.g. ``task_agent.py``) keep working.

Empirical: in a 10-gen polyglot run with pure haiku-4.5 (task + meta), this
path reaches 9/10 on the staged-eval JS subset by gen 7. The upstream prose
path stays at 0/10 across all 10 gens because the meta-agent never identifies
the load-bearing change (``tools_available='all'`` on ``task_agent.py``).
"""
import json

import litellm

from agent.llm import _extract_response_info, _openai_reasoning_effort
from agent.tools import load_tools

MAX_TOKENS = 16384
litellm.drop_params = True


def log_llm_usage(logging, info):
    if not info:
        return
    logging(f"LLM_USAGE: {json.dumps(info, sort_keys=True)}")


def chat_with_agent(
    msg,
    model="claude-4-sonnet-genai",
    msg_history=None,
    logging=print,
    tools_available=[],   # Empty list means no tools, 'all' means all tools
    multiple_tool_calls=False,  # Whether to allow multiple tool calls per turn
    max_tool_calls=40,    # Maximum tool calls allowed (-1 for unlimited)
):
    if msg_history is None:
        msg_history = []

    # Load HA tools and translate to OpenAI/litellm tool spec. Litellm
    # auto-normalizes this to Anthropic's {name, description, input_schema}
    # shape on the wire.
    ha_tools = load_tools(logging=logging, names=tools_available)
    tools_dict = {t["info"]["name"]: t for t in ha_tools}
    litellm_tools = [
        {
            "type": "function",
            "function": {
                "name": t["info"]["name"],
                "description": t["info"]["description"],
                "parameters": t["info"]["input_schema"],
            },
        }
        for t in ha_tools
    ]

    # Litellm uses OpenAI-style messages. HA's existing history is
    # [{role, text}, ...]; convert to {role, content}.
    messages = []
    for m in msg_history:
        messages.append({"role": m["role"], "content": m.get("text", m.get("content", ""))})
    messages.append({"role": "user", "content": msg})

    logging(f"Input: {repr(msg)}")
    num_tool_calls = 0

    try:
        while True:
            kwargs = {"model": model, "messages": messages}
            # Reasoning chat models (gpt-5*, o-series) use max_completion_tokens
            # and accept reasoning_effort. Legacy claude-3-haiku is capped at
            # 4096 output tokens. Everything else uses max_tokens.
            reasoning_effort = _openai_reasoning_effort(model)
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            if "gpt-5" in model or model.startswith(("openai/o", "o")):
                kwargs["max_completion_tokens"] = MAX_TOKENS
            elif "claude-3-haiku" in model:
                kwargs["max_tokens"] = min(MAX_TOKENS, 4096)
            else:
                kwargs["max_tokens"] = MAX_TOKENS

            if litellm_tools:
                kwargs["tools"] = litellm_tools
                kwargs["tool_choice"] = "auto"
                if not multiple_tool_calls:
                    kwargs["parallel_tool_calls"] = False

            response = litellm.completion(**kwargs)
            msg_obj = response.choices[0].message
            finish = response.choices[0].finish_reason
            logging(f"Output: {repr(msg_obj.content)}  finish_reason={finish}")
            log_llm_usage(logging, _extract_response_info(response, model))

            # Append the assistant turn (with tool_calls if any) so the next
            # call sees the full conversation context.
            assistant_entry = {"role": "assistant", "content": msg_obj.content or ""}
            if msg_obj.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg_obj.tool_calls
                ]
            messages.append(assistant_entry)

            # Loop exits when the model produces no tool_calls or signals
            # finish_reason != 'tool_calls'.
            if not msg_obj.tool_calls or finish != "tool_calls":
                break
            if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
                logging("Error: Maximum number of tool calls reached.")
                break

            # Execute each tool call and append a 'tool' role message per call.
            tool_calls = msg_obj.tool_calls if multiple_tool_calls else msg_obj.tool_calls[:1]
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    tool_input = {}
                if tool_name in tools_dict:
                    try:
                        out = tools_dict[tool_name]["function"](**tool_input)
                    except Exception as e:
                        out = f"Error executing tool '{tool_name}': {e}"
                else:
                    out = f"Error: Tool '{tool_name}' not found"
                if not isinstance(out, str):
                    out = str(out)
                logging(f"Tool output: {repr(out)}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
                num_tool_calls += 1
    except Exception as e:
        logging(f"Error: {str(e)}")
        raise

    # Convert back to HA's [{role, text}] shape so callers like
    # task_agent.py:extract_jsons keep working unchanged.
    new_msg_history = []
    for m in messages:
        text = m.get("content") or ""
        if m.get("tool_calls"):
            tc_strs = [
                f'<tool_use name="{c["function"]["name"]}">{c["function"]["arguments"]}</tool_use>'
                for c in m["tool_calls"]
            ]
            text = (text + "\n" + "\n".join(tc_strs)).strip()
        new_msg_history.append({"role": m["role"], "text": text})

    # Synthesize a terminal <json>{"response": "<final_text>"}</json> entry
    # so utils/common.py:extract_jsons (called from task_agent.py) finds the
    # expected wrapper unchanged.
    final_text = ""
    for m in reversed(new_msg_history):
        if m["role"] == "assistant" and m["text"]:
            final_text = m["text"]
            break
    new_msg_history.append({
        "role": "assistant",
        "text": f'<json>\n{json.dumps({"response": final_text})}\n</json>',
    })
    return new_msg_history


if __name__ == "__main__":
    msg = "hello"
    chat_with_agent(msg)
