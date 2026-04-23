import json
import re

from agent.llm import get_response_from_llm
from agent.tools import load_tools
from utils.common import extract_jsons

EDITOR_COMMANDS = {"view", "create", "str_replace", "insert", "undo_edit"}
MAX_TOOL_FORMAT_RETRIES = 3


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

    function_call_blocks = re.findall(r"<function_calls>(.*?)</function_calls>", response, re.DOTALL) or [response]
    for block in function_call_blocks:
        for tool_name, body in re.findall(r'<invoke name="([^"]+)">(.*?)</invoke>', block, re.DOTALL):
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
            block,
            re.DOTALL,
        ):
            parsed_tool_input = _parse_tool_input(tool_input.strip())
            if parsed_tool_input is None:
                continue
            tool_uses.append({"tool_name": tool_name.strip(), "tool_input": parsed_tool_input})

    return tool_uses


def should_retry_missing_tool_use(
    response,
    tool_uses=None,
    tool_infos=None,
    tool_call_count=0,
    require_tool_call_before_final=False,
):
    if not tool_infos or tool_uses:
        return False
    extracted_jsons = extract_jsons(response) or []
    if any(isinstance(item, dict) for item in extracted_jsons):
        return require_tool_call_before_final and tool_call_count == 0
    return bool(response.strip())


def check_for_tool_uses(response):
    """
    Checks if the response contains one or more tool calls in json code blocks.
    Returns a list of tool use dictionaries.
    """
    tool_uses = []
    extracted_jsons = extract_jsons(response) or []
    allow_editor_recovery = _looks_like_structured_json_block(response)
    for tool_use in extracted_jsons:
        if _is_valid_tool_use(tool_use):
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
            # Conservative recovery for a common malformed editor call:
            # {"command": "view", "path": "/testbed/grid_summary.md"}.
            if tool_use.get("command") in EDITOR_COMMANDS and isinstance(tool_use.get("path"), str):
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
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


def chat_with_agent(
    msg,
    model="claude-4-sonnet-genai",
    msg_history=None,
    logging=print,
    tools_available=[],  # Empty list means no tools, 'all' means all tools
    multiple_tool_calls=False,  # Whether to allow multiple tool calls in a single response
    max_tool_calls=40,  # Maximum number of tool calls allowed in a single response, -1 for unlimited
    return_on_error=False,  # Return partial history instead of raising provider/tool-loop errors
    require_tool_call_before_final=False,  # Keep retrying if the model returns final JSON before using any tool
):
    get_response_fn = get_response_from_llm
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history

    try:
        # Load all tools
        all_tools = load_tools(logging=logging, names=tools_available)
        tools_dict = {tool["info"]["name"]: tool for tool in all_tools}
        system_msg = f"{get_tooluse_prompt([tool['info'] for tool in all_tools])}\n\n"
        num_tool_calls = 0
        tool_format_retries = 0

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
        retry_missing_tool_use = should_retry_missing_tool_use(
            response,
            tool_uses,
            all_tools,
            tool_call_count=num_tool_calls,
            require_tool_call_before_final=require_tool_call_before_final,
        )
        while tool_uses or retry_tool_use or retry_missing_tool_use:
            # Check for max tool calls
            if max_tool_calls == 0 and tool_uses:
                raise RuntimeError("Maximum number of tool calls reached before a valid final response.")
            if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
                raise RuntimeError("Maximum number of tool calls reached before a valid final response.")
            if retry_tool_use or retry_missing_tool_use:
                tool_format_retries += 1
                if tool_format_retries > MAX_TOOL_FORMAT_RETRIES:
                    raise RuntimeError("Maximum number of malformed tool-response retries reached.")

            tool_msgs = []

            # Process tool uses
            if tool_uses:
                tool_format_retries = 0
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

            # Check for retry
            if retry_tool_use:
                logging("Error: Output context exceeded. Please try again.")
                tool_msgs.append("Error: Output context exceeded. Please try again.")
            if retry_missing_tool_use:
                logging("Error: Response must include one valid tool call or final JSON.")
                tool_msgs.append(
                    "Error: Response must include exactly one valid tool call in the required <json> format, "
                    "or a final <json> response if the task is complete."
                )

            # Get tool response
            response, new_msg_history, info = get_response_fn(
                msg=system_msg + "\n\n".join(tool_msgs),
                model=model,
                msg_history=new_msg_history,
            )
            log_llm_usage(logging, info)
            logging(f"Output: {repr(response)}")
            # logging(f"Info: {repr(info)}")

            # Check for next tool use
            tool_uses = check_for_tool_uses(response)
            retry_tool_use = should_retry_tool_use(response, tool_uses)
            retry_missing_tool_use = should_retry_missing_tool_use(
                response,
                tool_uses,
                all_tools,
                tool_call_count=num_tool_calls,
                require_tool_call_before_final=require_tool_call_before_final,
            )

    except Exception as e:
        logging(f"Error: {str(e)}")
        if return_on_error:
            return new_msg_history
        raise e

    return new_msg_history


if __name__ == "__main__":
    msg = """hello"""
    new_msg_history = chat_with_agent(msg)
