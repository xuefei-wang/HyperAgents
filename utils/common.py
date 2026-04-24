import re
import json
import os


def read_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

_FUNCTION_CALLS_INVOKE_RE = re.compile(
    r'<invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</invoke>',
    re.DOTALL,
)
_FUNCTION_CALLS_PARAM_RE = re.compile(
    r'<parameter\s+name="(?P<key>[^"]+)"\s*>(?P<value>.*?)</parameter>',
    re.DOTALL,
)


def _coerce_param_value(raw):
    """Best-effort type coercion for parameter values scraped from XML.

    Claude CLI-style tool-call XML (``<parameter name="X">VALUE</parameter>``)
    does not commit to a value encoding. VALUE is usually a plain string or a
    JSON literal; occasionally it is a JSON fragment with stray braces. Try
    JSON first; fall back to the trimmed raw string.
    """
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _extract_function_calls_xml(response):
    """Parse Anthropic CLI-style ``<function_calls><invoke>...`` tool calls.

    Claude Haiku 4.5 (and other post-2025 Anthropic models) spontaneously emit
    this format when they infer tool-access context, even when the system
    prompt requests a different convention. Upstream HyperAgents never
    registered this dialect in ``extract_jsons`` — the model's tool calls were
    silently discarded, the tool loop exited after one turn, and every task
    produced a 0-byte ``model_patch.diff``.

    Match each ``<invoke name="X">...<parameter name="Y">VAL</parameter>...
    </invoke>`` and materialise ``{"tool_name": "X", "tool_input": {"Y": VAL,
    ...}}`` so the existing downstream tool loop consumes it unchanged.
    """
    results = []
    for invoke_match in _FUNCTION_CALLS_INVOKE_RE.finditer(response):
        tool_name = invoke_match.group("name").strip()
        if not tool_name:
            continue
        tool_input = {}
        for param_match in _FUNCTION_CALLS_PARAM_RE.finditer(invoke_match.group("body")):
            tool_input[param_match.group("key").strip()] = _coerce_param_value(
                param_match.group("value")
            )
        # Haiku 4.5 has been observed to merge HA's ``tool_input`` convention
        # into its native XML format by emitting a single
        # ``<parameter name="tool_input">{...}</parameter>`` whose value is
        # already the schema-shaped input dict. Unwrap that single layer so
        # downstream sees ``{"tool_name": "bash", "tool_input": {"command": ...}}``
        # instead of ``{"tool_name": "bash", "tool_input": {"tool_input": {...}}}``.
        if (
            len(tool_input) == 1
            and "tool_input" in tool_input
            and isinstance(tool_input["tool_input"], dict)
        ):
            tool_input = tool_input["tool_input"]
        # A bare ``<invoke>`` with no parameters still counts — upstream tools
        # like ``list_tools`` take zero args. Emit the dict either way so the
        # caller can decide.
        results.append({"tool_name": tool_name, "tool_input": tool_input})
    return results


def extract_jsons(response):
    """
    Extracts all JSON objects from the given response string.

    Supports three encodings the HA tool loop might receive:
      1. ``<json>{...}</json>`` — HA's own system-prompted format
      2. Triple-backtick ``json`` fenced blocks
      3. ``<function_calls><invoke name="...">...</invoke></function_calls>``
         — Claude's native tool-call dialect that newer Anthropic models
         (Haiku 4.5, Sonnet 4.5+) emit spontaneously
    Falls back to greedy JSON-object decoding if none of the above match.
    """
    patterns = [
        r'<json>\s*(\{.*?\})\s*(?:</json>)?',
        r'```json(.*?)```',
    ]
    extracted_jsons = []

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                extracted_json = json.loads(match.strip())
                extracted_jsons.append(extracted_json)
            except json.JSONDecodeError:
                continue  # Skip malformed JSON blocks

    # Fold in Anthropic CLI-style function_calls, if present. These are
    # already normalised to ``{"tool_name", "tool_input"}`` dicts.
    extracted_jsons.extend(_extract_function_calls_xml(response))

    if extracted_jsons:
        return extracted_jsons

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(response):
        if response[idx] != "{":
            idx += 1
            continue
        try:
            parsed, end = decoder.raw_decode(response[idx:])
            if isinstance(parsed, dict):
                extracted_jsons.append(parsed)
            idx += end
        except json.JSONDecodeError:
            idx += 1

    return extracted_jsons if extracted_jsons else None

def file_exist_and_not_empty(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def load_json_file(file_path):
    """
    Load a JSON file and return its contents as a dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)
