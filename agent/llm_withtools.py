import json
import re

import litellm

from agent.llm import (
    _extract_response_info,
    _is_openai_reasoning_model,
    _openai_reasoning_effort,
    _supports_custom_temperature,
    get_response_from_llm,
)
from agent.tools import load_tools
from utils.common import extract_jsons


def log_llm_usage(logging, info):
    if not info:
        return
    logging(f"LLM_USAGE: {json.dumps(info, sort_keys=True)}")


def get_tooluse_prompt(tool_infos=[]):
    """
    Get the prompt for using the available tools.
    """
    if not tool_infos or len(tool_infos) == 0:
        return ""

    tools_available = [str(tool_info) for tool_info in tool_infos]
    tools_available = "\n\n".join(tools_available) if tools_available else "None"
    tooluse_prompt = """Here are the available tools:
```
{tools_available}
```

Use at most one tool per assistant response. You may use tools repeatedly across multiple responses:
after each tool result, decide the next single tool call or provide the final answer.

Use tools in this format:
<json>
{{
    "tool_name": ...,
    "tool_input": ...
}}
</json>

`tool_input` must always be a JSON object matching the tool schema, not a string. For example, a bash call must be:
<json>
{{
    "tool_name": "bash",
    "tool_input": {{"command": "cd /testbed && ls"}}
}}
</json>

ONLY USE ONE TOOL PER RESPONSE, BUT KEEP USING ONE TOOL AT A TIME UNTIL THE TASK IS DONE.
STRICTLY FOLLOW THE FORMAT OF TOOL_NAME AND TOOL_INPUT ABOVE.
DO NOT HALLUCINATE OR MAKE UP ANYTHING.
""".format(tools_available=tools_available)
    return tooluse_prompt.strip()


def should_retry_tool_use(response, tool_uses=None):
    """
    Check if the response attempts to use a tool,
    but ran out of output context.
    """
    if tool_uses is not None and len(tool_uses) > 0:
        return False

    json_pos = response.find("<json>")
    tool_name_pos = response.find("tool_name")
    tool_input_pos = response.find("tool_input")

    if (
        json_pos != -1
        and tool_name_pos != -1
        and tool_input_pos != -1
        and json_pos < tool_name_pos < tool_input_pos
        and len(response) >= 2000
    ):
        return True

    return False


def _parse_tool_input(raw_input):
    if isinstance(raw_input, dict):
        return raw_input
    if not isinstance(raw_input, str):
        return None
    parsed_jsons = extract_jsons(raw_input) or []
    for parsed in parsed_jsons:
        if isinstance(parsed, dict):
            return parsed
    try:
        parsed = json.loads(raw_input)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _looks_like_structured_json_block(response):
    return "<json>" in response or "```json" in response


def _is_valid_tool_use(tool_use):
    return (
        isinstance(tool_use, dict)
        and set(tool_use.keys()) == {"tool_name", "tool_input"}
        and isinstance(tool_use.get("tool_name"), str)
        and isinstance(tool_use.get("tool_input"), dict)
    )


def _extract_function_call_uses(response):
    tool_uses = []

    for tool_name, body in re.findall(r'<invoke name="([^"]+)">(.*?)</invoke>', response, re.DOTALL):
        params = {
            name: value.strip()
            for name, value in re.findall(r'<parameter name="([^"]+)">(.*?)</parameter>', body, re.DOTALL)
        }
        if not params:
            continue
        tool_input = params
        if "tool_input" in params:
            tool_input = _parse_tool_input(params["tool_input"])
            if tool_input is None:
                continue
        tool_uses.append({"tool_name": tool_name.strip(), "tool_input": tool_input})

    for tool_name, tool_input in re.findall(
        r'<function_name>([^<]+)</(?:function_name|parameter)>\s*<parameter name="tool_input">(.*?)</parameter>',
        response,
        re.DOTALL,
    ):
        parsed_tool_input = _parse_tool_input(tool_input.strip())
        if parsed_tool_input is None:
            continue
        tool_uses.append({"tool_name": tool_name.strip(), "tool_input": parsed_tool_input})

    return tool_uses


def should_retry_missing_tool_use(response, tool_uses=None, tool_infos=None):
    if not tool_infos or tool_uses:
        return False
    extracted_jsons = extract_jsons(response) or []
    if any(isinstance(item, dict) for item in extracted_jsons):
        return False
    return bool(response.strip())


def check_for_tool_uses(response):
    """
    Checks if the response contains one or more tool calls in json code blocks.
    Returns a list of tool use dictionaries.
    """
    tool_uses = []
    stripped_response = response.strip()
    allow_raw_json_tool_use = stripped_response.startswith("{") and stripped_response.endswith("}")
    allow_structured_tool_use = _looks_like_structured_json_block(response) or allow_raw_json_tool_use
    extracted_jsons = extract_jsons(response) or []
    allow_editor_recovery = allow_structured_tool_use

    for tool_use in extracted_jsons:
        if _is_valid_tool_use(tool_use):
            if not allow_structured_tool_use:
                continue
            tool_uses.append(
                {
                    "tool_name": tool_use["tool_name"].strip(),
                    "tool_input": tool_use["tool_input"],
                }
            )
            continue
        if not isinstance(tool_use, dict):
            continue
        if allow_editor_recovery and "tool_name" not in tool_use and "tool_input" not in tool_use:
            if tool_use.get("command") in {"view", "create", "str_replace", "insert", "undo_edit"} and isinstance(
                tool_use.get("path"), str
            ):
                tool_uses.append({"tool_name": "editor", "tool_input": tool_use})

    tool_uses.extend(_extract_function_call_uses(response))

    deduped_tool_uses = []
    seen = set()
    for tool_use in tool_uses:
        key = json.dumps(tool_use, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduped_tool_uses.append(tool_use)

    return deduped_tool_uses if deduped_tool_uses else None


def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]["function"](**tool_input)
        return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


def supports_native_tool_calling(model):
    normalized = (model or "").lower()
    return (
        normalized.startswith("anthropic/")
        or normalized.startswith("openai/")
        or normalized.startswith(("gpt-", "o1", "o3", "o4"))
    )


def convert_tool_info(tool_info):
    return {
        "type": "function",
        "function": {
            "name": tool_info["name"],
            "description": tool_info["description"],
            "parameters": tool_info["input_schema"],
        },
    }


def _message_content(message):
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _message_tool_calls(message):
    if isinstance(message, dict):
        return message.get("tool_calls")
    return getattr(message, "tool_calls", None)


def extract_native_tool_calls(message):
    tool_calls = _message_tool_calls(message) or []
    normalized_calls = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            function = tool_call.get("function") or {}
            call_id = tool_call.get("id")
            name = function.get("name")
            arguments = function.get("arguments", "{}")
        else:
            function = getattr(tool_call, "function", None)
            call_id = getattr(tool_call, "id", None)
            name = getattr(function, "name", None) if function is not None else None
            arguments = getattr(function, "arguments", "{}") if function is not None else "{}"

        try:
            parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed_arguments, dict):
            continue

        normalized_calls.append(
            {
                "id": call_id or f"tool_call_{len(normalized_calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(parsed_arguments),
                },
            }
        )
    return normalized_calls


def to_native_messages(msg_history):
    messages = []
    for msg in msg_history:
        role = msg.get("role")
        if role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id"),
                    "content": msg.get("text", ""),
                }
            )
            continue

        native_message = {
            "role": role,
            "content": msg.get("text", ""),
        }
        if msg.get("tool_calls"):
            native_message["tool_calls"] = msg["tool_calls"]
            if native_message["content"] == "":
                native_message["content"] = None
        messages.append(native_message)
    return messages


def format_tool_output(tool_name, tool_input, tool_output):
    return f"""<json>
{{
    "tool_name": "{tool_name}",
    "tool_input": {tool_input},
    "tool_output": "{tool_output}"
}}
</json>""".strip()


def chat_with_agent_native(
    msg,
    model,
    msg_history,
    logging,
    all_tools,
    max_tool_calls=40,
):
    tools_dict = {tool["info"]["name"]: tool for tool in all_tools}
    tools = [convert_tool_info(tool["info"]) for tool in all_tools]
    new_msg_history = list(msg_history)
    new_msg_history.append({"role": "user", "text": msg})
    num_tool_calls = 0

    logging(f"Input: {repr(msg)}")

    while True:
        completion_kwargs = {
            "model": model,
            "messages": to_native_messages(new_msg_history),
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        if _supports_custom_temperature(model):
            completion_kwargs["temperature"] = 0.0
        if _is_openai_reasoning_model(model):
            completion_kwargs["max_completion_tokens"] = 4096
        else:
            completion_kwargs["max_tokens"] = 4096
        reasoning_effort = _openai_reasoning_effort(model)
        if reasoning_effort:
            completion_kwargs["reasoning_effort"] = reasoning_effort
        response = litellm.completion(**completion_kwargs)
        info = _extract_response_info(response, model)
        log_llm_usage(logging, info)
        message = response["choices"][0]["message"]
        response_text = _message_content(message) or ""
        tool_calls = extract_native_tool_calls(message)
        logging(f"Output: {repr(response_text if response_text else message)}")

        if not tool_calls:
            new_msg_history.append({"role": "assistant", "text": response_text})
            return new_msg_history

        if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
            logging("Error: Maximum number of tool calls reached.")
            new_msg_history.append(
                {
                    "role": "assistant",
                    "text": response_text or "Error: Maximum number of tool calls reached.",
                }
            )
            return new_msg_history

        tool_call = tool_calls[0]
        tool_name = tool_call["function"]["name"]
        tool_input = json.loads(tool_call["function"]["arguments"])
        tool_output = process_tool_call(tools_dict, tool_name, tool_input)
        num_tool_calls += 1

        new_msg_history.append(
            {
                "role": "assistant",
                "text": response_text,
                "tool_calls": tool_calls,
            }
        )
        new_msg_history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "text": str(tool_output),
            }
        )
        logging(f"Tool output: {repr(format_tool_output(tool_name, tool_input, tool_output))}")


def chat_with_agent(
    msg,
    model="claude-4-sonnet-genai",
    msg_history=None,
    logging=print,
    tools_available=[],
    multiple_tool_calls=False,
    max_tool_calls=40,
    return_on_error=False,
):
    get_response_fn = get_response_from_llm
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history

    try:
        all_tools = load_tools(logging=logging, names=tools_available)
        if all_tools and supports_native_tool_calling(model):
            return chat_with_agent_native(
                msg=msg,
                model=model,
                msg_history=msg_history,
                logging=logging,
                all_tools=all_tools,
                max_tool_calls=max_tool_calls,
            )

        tools_dict = {tool["info"]["name"]: tool for tool in all_tools}
        system_msg = f"{get_tooluse_prompt([tool['info'] for tool in all_tools])}\n\n"
        num_tool_calls = 0

        logging(f"Input: {repr(msg)}")
        response, new_msg_history, info = get_response_fn(
            msg=system_msg + msg,
            model=model,
            msg_history=new_msg_history,
        )
        log_llm_usage(logging, info)
        logging(f"Output: {repr(response)}")

        tool_uses = check_for_tool_uses(response)
        retry_tool_use = should_retry_tool_use(response, tool_uses)
        retry_missing_tool_use = should_retry_missing_tool_use(response, tool_uses, all_tools)
        while tool_uses or retry_tool_use or retry_missing_tool_use:
            if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
                logging("Error: Maximum number of tool calls reached.")
                break

            tool_msgs = []

            if tool_uses:
                tool_uses = tool_uses if multiple_tool_calls else tool_uses[:1]
                for tool_use in tool_uses:
                    tool_name = tool_use["tool_name"]
                    tool_input = tool_use["tool_input"]
                    tool_output = process_tool_call(tools_dict, tool_name, tool_input)
                    num_tool_calls += 1
                    tool_payload = json.dumps(
                        {
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "tool_output": tool_output,
                        },
                        ensure_ascii=False,
                    )
                    tool_msg = f"<json>\n{tool_payload}\n</json>"
                    logging(f"Tool output: {repr(tool_msg)}")
                    tool_msgs.append(tool_msg)

            if retry_tool_use:
                logging("Error: Output context exceeded. Please try again.")
                tool_msgs.append("Error: Output context exceeded. Please try again.")
            if retry_missing_tool_use:
                logging("Error: Response must include one valid tool call or final JSON.")
                tool_msgs.append(
                    "Error: Response must include exactly one valid tool call in the required <json> format, "
                    "or a final <json> response if the task is complete."
                )

            response, new_msg_history, info = get_response_fn(
                msg=system_msg + "\n\n".join(tool_msgs),
                model=model,
                msg_history=new_msg_history,
            )
            log_llm_usage(logging, info)
            logging(f"Output: {repr(response)}")

            tool_uses = check_for_tool_uses(response)
            retry_tool_use = should_retry_tool_use(response, tool_uses)
            retry_missing_tool_use = should_retry_missing_tool_use(response, tool_uses, all_tools)

    except Exception as e:
        logging(f"Error: {str(e)}")
        if return_on_error:
            return new_msg_history
        raise e

    return new_msg_history


if __name__ == "__main__":
    msg = """hello"""
    new_msg_history = chat_with_agent(msg)
