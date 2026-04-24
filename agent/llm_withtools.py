import re
import json

from agent.llm import get_response_from_llm
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

You may use one or more tools per assistant response. When you emit multiple
tool calls in a single response, they are executed sequentially in the order
written and the results are returned together before your next turn.

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

_TOOL_SHAPED_SIGNALS = ("<json>", "<function_calls>", "<invoke ", "```json", '"tool_name"')


def check_for_tool_uses(response, logging=None):
    """
    Checks if the response contains one or more tool calls in json code blocks.
    Returns a list of tool use dictionaries.

    If the response LOOKS like it was trying to invoke tools but nothing
    parsed, emit a "TOOL_PARSER_MISS" log line via ``logging`` (default: silent
    for backward compatibility). Silent misses were the root cause of both the
    Haiku #488 degenerate-text loop and the gpt-5.4-mini multi-stanza drop —
    parser returns None, tool loop exits, failure shows up only as a 0-byte
    patch at the end of the run.
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

    if not tool_uses and logging is not None:
        if any(signal in response for signal in _TOOL_SHAPED_SIGNALS):
            logging(
                "TOOL_PARSER_MISS: response contains tool-shaped markers but no "
                "tool call was parsed. Investigate extract_jsons regex coverage."
            )

    return tool_uses if tool_uses else None

def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
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
    multiple_tool_calls=True,  # Allow multiple tool calls in one response. Reasoning-heavy
    # models (gpt-5.4-mini, o-series) routinely plan N tools per turn; the
    # prior default of False silently dropped every stanza after the first,
    # so the agent re-emitted the same exploration in the next turn and
    # never advanced. See failure-mode audit on audit_sweep_openai_openai_audit.
    max_tool_calls=40,  # Maximum number of tool calls allowed in a single response, -1 for unlimited
    return_on_error=False,  # Return partial history instead of raising provider/tool-loop errors
):
    get_response_fn = get_response_from_llm
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history

    try:
        # Load all tools
        all_tools = load_tools(logging=logging, names=tools_available)
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
        tool_uses = check_for_tool_uses(response, logging=logging)
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
            tool_uses = check_for_tool_uses(response, logging=logging)
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
