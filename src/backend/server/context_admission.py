"""Exact static context budgeting for OpenAI-compatible chat requests."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping

DEFAULT_MAX_OUTPUT_TOKENS = 128


class ContextRequestError(ValueError):
    """The request cannot be tokenized or contains an invalid token budget."""


@dataclass(frozen=True)
class ContextBudget:
    prompt_tokens: int
    max_output_tokens: int

    @property
    def required_tokens(self) -> int:
        return self.prompt_tokens + self.max_output_tokens


def _max_output_tokens(request_data: Mapping[str, Any]) -> int:
    value = request_data.get("max_completion_tokens")
    if value is None:
        value = request_data.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContextRequestError("max_completion_tokens/max_tokens must be a positive integer")
    return value


def _template_kwargs(request_data: Mapping[str, Any]) -> Dict[str, Any]:
    raw_kwargs = request_data.get("chat_template_kwargs")
    if raw_kwargs is None:
        kwargs: Dict[str, Any] = {}
    elif isinstance(raw_kwargs, dict):
        kwargs = dict(raw_kwargs)
    else:
        raise ContextRequestError("chat_template_kwargs must be a JSON object")

    for name in ("tools", "documents"):
        value = request_data.get(name)
        if value is not None:
            kwargs.setdefault(name, value)

    chat_template = request_data.get("chat_template")
    if chat_template is not None:
        if not isinstance(chat_template, str):
            raise ContextRequestError("chat_template must be a string")
        kwargs["chat_template"] = chat_template

    continue_final_message = request_data.get("continue_final_message", False)
    if not isinstance(continue_final_message, bool):
        raise ContextRequestError("continue_final_message must be a boolean")
    add_generation_prompt = request_data.get("add_generation_prompt", not continue_final_message)
    if not isinstance(add_generation_prompt, bool):
        raise ContextRequestError("add_generation_prompt must be a boolean")
    if continue_final_message and add_generation_prompt:
        raise ContextRequestError(
            "add_generation_prompt and continue_final_message cannot both be true"
        )

    kwargs["tokenize"] = True
    kwargs["add_generation_prompt"] = add_generation_prompt
    if continue_final_message:
        kwargs["continue_final_message"] = True
    return kwargs


def _token_count(tokenized: Any) -> int:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized.get("input_ids")
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if not isinstance(tokenized, list):
        raise ContextRequestError("chat tokenizer did not return token ids")
    if tokenized and isinstance(tokenized[0], list):
        if len(tokenized) != 1:
            raise ContextRequestError("chat tokenizer returned an unexpected batch")
        tokenized = tokenized[0]
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in tokenized):
        raise ContextRequestError("chat tokenizer returned invalid token ids")
    return len(tokenized)


def build_context_budget(tokenizer: Any, request_data: Mapping[str, Any]) -> ContextBudget:
    """Render and tokenize the final chat prompt using the model's own template."""
    max_output_tokens = _max_output_tokens(request_data)
    messages = request_data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContextRequestError("messages must be a non-empty array")
    if any(not isinstance(message, dict) for message in messages):
        raise ContextRequestError("each message must be a JSON object")

    try:
        tokenized = tokenizer.apply_chat_template(messages, **_template_kwargs(request_data))
    except ContextRequestError:
        raise
    except Exception as exc:
        raise ContextRequestError(f"unable to render chat template: {exc}") from exc

    return ContextBudget(
        prompt_tokens=_token_count(tokenized),
        max_output_tokens=max_output_tokens,
    )
