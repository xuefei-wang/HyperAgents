import re
import json

from agent.llm import get_response_from_llm
from agent.tools import load_tools


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

Use only one tool (if needed) in this format:
<json>
{{
    "tool_name": ...,
    "tool_input": ...
}}
</json>

ONLY USE ONE TOOL PER RESPONSE, AND STRICTLY FOLLOW THE FORMAT OF TOOL_NAME AND TOOL_INPUT ABOVE.
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

# HyperAgents' documented protocol is <json>{tool_name,tool_input}</json>, but
# Claude 4.5-class models (Haiku 4.5) emit their native <function_calls>/<invoke>
# tool XML instead -- either a <json> block closed by </function_calls>, or a
# pure <invoke name=...><parameter name=...>...</invoke> block with no <json> at
# all. The strict <json>...</json> regex dropped every such call, so no tool ever
# executed. Accept all three forms.
_JSON_TOOL_RE = re.compile(r'<json>\s*(\{.*?\})\s*</(?:json|function_calls)>', re.DOTALL)
_INVOKE_RE = re.compile(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.DOTALL)
_PARAM_RE = re.compile(r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>', re.DOTALL)


def _parse_json_tool(blob):
    try:
        tool_use = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(tool_use, dict) or 'tool_name' not in tool_use or 'tool_input' not in tool_use:
        return None
    return tool_use


def _parse_invoke_tool(name, body):
    tool_input = {}
    for pm in _PARAM_RE.finditer(body):
        tool_input[pm.group(1)] = pm.group(2).strip()
    return {'tool_name': name, 'tool_input': tool_input}


def check_for_tool_uses(response):
    """
    Return the response's tool calls in emission order as a list of
    ``{"tool_name", "tool_input"}`` dicts, or ``None`` if there are none.

    Accepts the ``<json>{...}</json>`` protocol form, the Haiku hybrid where the
    ``<json>`` block is closed by ``</function_calls>``, and the pure native
    ``<invoke name=...><parameter name=...>...</invoke>`` form. Ordering is
    preserved because the tool loop consumes only the first call by default.
    """
    found = []  # (position, tool_use)
    for m in _JSON_TOOL_RE.finditer(response):
        tool_use = _parse_json_tool(m.group(1))
        if tool_use is not None:
            found.append((m.start(), tool_use))
    for m in _INVOKE_RE.finditer(response):
        found.append((m.start(), _parse_invoke_tool(m.group(1), m.group(2))))
    found.sort(key=lambda pair: pair[0])
    tool_uses = [tool_use for _, tool_use in found]
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
    multiple_tool_calls=False,  # Whether to allow multiple tool calls in a single response
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
