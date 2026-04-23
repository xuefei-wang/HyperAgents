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
    # If no tools are available, return an empty string
    if not tool_infos or len(tool_infos) == 0:
        return ""
    # Create the prompt
    tools_available = [str(tool_info) for tool_info in tool_infos]
    tools_available = '\n\n'.join(tools_available) if tools_available else 'None'
    tooluse_prompt = """Here are the available tools:
```
{tools_available}
```

Use at most one tool per assistant response. You may use tools repeatedly across multiple responses: after each tool result, decide the next single tool call or provide the final answer.

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
    # If there are tool uses, we don't need to check for retry
    if tool_uses is not None and len(tool_uses) > 0:
        return False

    # Find positions of the markers
    json_pos = response.find("<json>")
    tool_name_pos = response.find("tool_name")
    tool_input_pos = response.find("tool_input")

    # Check ordering and length condition
    if (
        json_pos != -1
        and tool_name_pos != -1
        and tool_input_pos != -1
        and json_pos < tool_name_pos < tool_input_pos
        and len(response) >= 2000
    ):
        return True

    # No retry
    return False

def check_for_tool_uses(response):
    """
    Checks if the response contains one or more tool calls in json code blocks.
    Returns a list of tool use dictionaries.
    """
    tool_uses = []
    extracted_jsons = extract_jsons(response) or []
    for tool_use in extracted_jsons:
        if not isinstance(tool_use, dict):
            continue
        if 'tool_name' not in tool_use or 'tool_input' not in tool_use:
            # Conservative recovery for a common malformed editor call:
            # {"command": "view", "path": "/testbed/grid_summary.md"}.
            if (
                'tool_name' not in tool_use
                and 'tool_input' not in tool_use
                and tool_use.get('command') in {'view', 'create', 'str_replace', 'insert', 'undo_edit'}
                and isinstance(tool_use.get('path'), str)
            ):
                tool_use = {'tool_name': 'editor', 'tool_input': tool_use}
            else:
                continue
        tool_uses.append(tool_use)

    return tool_uses if tool_uses else None

def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
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
    return f'''<json>
    {{
        "tool_name": "{tool_name}",
        "tool_input": {tool_input},
        "tool_output": "{tool_output}"
    }}
    </json>'''.strip()


def chat_with_agent_native(
    msg,
    model,
    msg_history,
    logging,
    all_tools,
    max_tool_calls=40,
):
    tools_dict = {tool['info']['name']: tool for tool in all_tools}
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
    tools_available=[],  # Empty list means no tools, 'all' means all tools
    multiple_tool_calls=False,  # Whether to allow multiple tool calls in a single response
    max_tool_calls=40,  # Maximum number of tool calls allowed in a single response, -1 for unlimited
    return_on_error=False,  # Return partial history instead of raising provider/tool-loop errors
):
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

        get_response_fn = get_response_from_llm
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        system_msg = f"{get_tooluse_prompt([tool['info'] for tool in all_tools])}\n\n"
        num_tool_calls = 0

        # Call API
        logging(f"Input: {repr(msg)}")
        response, new_msg_history, info = get_response_fn(
            msg=system_msg + msg,
            model=model,
            msg_history=new_msg_history,
        )
        log_llm_usage(logging, info)
        logging(f"Output: {repr(response)}")
        # logging(f"Info: {repr(info)}")

        # Tool use
        tool_uses = check_for_tool_uses(response)
        retry_tool_use = should_retry_tool_use(response, tool_uses)
        while tool_uses or retry_tool_use:
            # Check for max tool calls
            if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
                logging("Error: Maximum number of tool calls reached.")
                break

            tool_msgs = []

            # Process tool uses
            if tool_uses:
                tool_uses = tool_uses if multiple_tool_calls else tool_uses[:1]
                for tool_use in tool_uses:
                    tool_name = tool_use['tool_name']
                    tool_input = tool_use['tool_input']
                    tool_output = process_tool_call(tools_dict, tool_name, tool_input)
                    num_tool_calls += 1
                    tool_msg = f'''<json>
    {{
        "tool_name": "{tool_name}",
        "tool_input": {tool_input},
        "tool_output": "{tool_output}"
    }}
    </json>'''.strip()
                    logging(f"Tool output: {repr(tool_msg)}")
                    tool_msgs.append(tool_msg)

            # Check for retry
            if retry_tool_use:
                logging("Error: Output context exceeded. Please try again.")
                tool_msgs.append("Error: Output context exceeded. Please try again.")

            # Get tool response
            response, new_msg_history, info = get_response_fn(
                msg=system_msg + '\n\n'.join(tool_msgs),
                model=model,
                msg_history=new_msg_history,
            )
            log_llm_usage(logging, info)
            logging(f"Output: {repr(response)}")
            # logging(f"Info: {repr(info)}")

            # Check for next tool use
            tool_uses = check_for_tool_uses(response)
            retry_tool_use = should_retry_tool_use(response, tool_uses)

    except Exception as e:
        logging(f"Error: {str(e)}")
        if return_on_error:
            return new_msg_history
        raise e

    return new_msg_history

if __name__ == "__main__":
    msg = """hello"""
    new_msg_history = chat_with_agent(msg)
