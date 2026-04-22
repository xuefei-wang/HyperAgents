import backoff
import os
from pathlib import Path
from typing import Tuple
import requests
import litellm
from dotenv import load_dotenv
import json


def _load_shared_env() -> None:
    llm_path = Path(__file__).resolve()
    candidate_roots = [
        llm_path.parents[1],  # HyperAgents repo root, if configs are copied in.
        llm_path.parents[2],  # Parent baseline directory.
        llm_path.parents[3],  # Repository root when HyperAgents lives under baselines/.
        Path.cwd(),
    ]
    loaded = set()
    for repo_root in candidate_roots:
        for env_path in [
            repo_root / "configs" / "providers" / ".env.shared",
            repo_root / "configs" / "providers" / ".env.haiku",
            repo_root / "configs" / "providers" / ".env.openai",
            repo_root / "configs" / "models" / "shared.env",
        ]:
            if env_path in loaded:
                continue
            loaded.add(env_path)
            if env_path.exists():
                load_dotenv(env_path, override=True)


load_dotenv()
_load_shared_env()

MAX_TOKENS = 16384

CLAUDE_MODEL = "anthropic/claude-sonnet-4-5-20250929"
CLAUDE_HAIKU_MODEL = "anthropic/claude-3-haiku-20240307"
CLAUDE_35NEW_MODEL = "anthropic/claude-3-5-sonnet-20241022"
OPENAI_GPT54MINI_MODEL = "openai/gpt-5.4-mini"
OPENAI_MODEL = OPENAI_GPT54MINI_MODEL
OPENAI_MINI_MODEL = OPENAI_GPT54MINI_MODEL
OPENAI_O3_MODEL = "openai/o3"
OPENAI_O3MINI_MODEL = "openai/o3-mini"
OPENAI_O4MINI_MODEL = "openai/o4-mini"
OPENAI_GPT52_MODEL = "openai/gpt-5.2"
OPENAI_GPT5_MODEL = "openai/gpt-5"
OPENAI_GPT5MINI_MODEL = "openai/gpt-5-mini"
GEMINI_3_MODEL = "gemini/gemini-3-pro-preview"
GEMINI_MODEL = "gemini/gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini/gemini-2.5-flash"

DEFAULT_OPENAI_REASONING_EFFORT = "medium"

litellm.drop_params=True


def _openai_reasoning_effort(model):
    effort = (
        os.getenv("HYPERAGENTS_REASONING_EFFORT")
        or os.getenv("OPENAI_REASONING_EFFORT")
        or DEFAULT_OPENAI_REASONING_EFFORT
    ).strip()
    if not effort:
        return None

    model_name = _provider_model_name(model)
    if model_name.startswith(("gpt-5", "o1-", "o3-", "o4-")):
        return effort
    return None


def _provider_model_name(model):
    return model.split("/", 1)[-1].lower()


def _supports_custom_temperature(model):
    model_name = _provider_model_name(model)
    if not model_name.startswith("gpt-5"):
        return True
    return model_name.startswith("gpt-5.2")


def _to_plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    try:
        return dict(value)
    except Exception:
        return {}


def _extract_response_info(response, model):
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    usage_dict = _to_plain_dict(usage)

    cost = None
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None

    response_id = getattr(response, "id", None)
    if response_id is None and isinstance(response, dict):
        response_id = response.get("id")

    return {
        "model": model,
        "response_id": response_id,
        "usage": usage_dict,
        "cost_usd": cost,
    }

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
) -> Tuple[str, list, dict]:
    if msg_history is None:
        msg_history = []

    # Convert text to content, compatible with LITELLM API
    msg_history = [
        {**msg, "content": msg.pop("text")} if "text" in msg else msg
        for msg in msg_history
    ]

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # Build kwargs - handle model-specific requirements
    completion_kwargs = {
        "model": model,
        "messages": new_msg_history,
    }

    # GPT-5.2 supports temperature; earlier GPT-5 variants and GPT-5.4 use the default.
    if _supports_custom_temperature(model):
        completion_kwargs["temperature"] = temperature

    # GPT-5 models require max_completion_tokens instead of max_tokens
    if "gpt-5" in model:
        completion_kwargs["max_completion_tokens"] = max_tokens
    else:
        # Claude Haiku has a 4096 token limit
        if "claude-3-haiku" in model:
            completion_kwargs["max_tokens"] = min(max_tokens, 4096)
        else:
            completion_kwargs["max_tokens"] = max_tokens

    reasoning_effort = _openai_reasoning_effort(model)
    if reasoning_effort:
        completion_kwargs["reasoning_effort"] = reasoning_effort

    response = litellm.completion(**completion_kwargs)
    info = _extract_response_info(response, model)
    response_text = response['choices'][0]['message']['content']  # pyright: ignore
    new_msg_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

    # Convert content to text, compatible with MetaGen API
    new_msg_history = [
        {**msg, "text": msg.pop("content")} if "content" in msg else msg
        for msg in new_msg_history
    ]

    return response_text, new_msg_history, info


if __name__ == "__main__":
    msg = 'Hello there!'
    models = [
        ("CLAUDE_MODEL", CLAUDE_MODEL),
        ("CLAUDE_HAIKU_MODEL", CLAUDE_HAIKU_MODEL),
        ("CLAUDE_35NEW_MODEL", CLAUDE_35NEW_MODEL),
        ("OPENAI_MODEL", OPENAI_MODEL),
        ("OPENAI_MINI_MODEL", OPENAI_MINI_MODEL),
        ("OPENAI_O3_MODEL", OPENAI_O3_MODEL),
        ("OPENAI_O3MINI_MODEL", OPENAI_O3MINI_MODEL),
        ("OPENAI_O4MINI_MODEL", OPENAI_O4MINI_MODEL),
        ("OPENAI_GPT52_MODEL", OPENAI_GPT52_MODEL),
        ("OPENAI_GPT5_MODEL", OPENAI_GPT5_MODEL),
        ("OPENAI_GPT5MINI_MODEL", OPENAI_GPT5MINI_MODEL),
        ("OPENAI_GPT54MINI_MODEL", OPENAI_GPT54MINI_MODEL),
        ("GEMINI_3_MODEL", GEMINI_3_MODEL),
        ("GEMINI_MODEL", GEMINI_MODEL),
        ("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL),
    ]
    for name, model in models:
        print(f"\n{'='*50}")
        print(f"Testing {name}: {model}")
        print('='*50)
        try:
            output_msg, msg_history, info = get_response_from_llm(msg, model=model)
            print(f"OK: {output_msg[:100]}...")
        except Exception as e:
            print(f"FAIL: {str(e)[:200]}")
