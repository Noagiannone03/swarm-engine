from collections import UserDict

import pytest

from backend.server.context_admission import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    ContextRequestError,
    build_context_budget,
)


class RecordingTokenizer:
    def __init__(self, result=None, error=None):
        self.result = list(range(17)) if result is None else result
        self.error = error
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


def test_budget_uses_rendered_chat_tools_and_reserved_output():
    tokenizer = RecordingTokenizer()
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Inspect the repository."},
    ]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    budget = build_context_budget(
        tokenizer,
        {
            "messages": messages,
            "tools": tools,
            "documents": [{"title": "handoff", "text": "..."}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_completion_tokens": 2048,
        },
    )

    assert budget.prompt_tokens == 17
    assert budget.max_output_tokens == 2048
    assert budget.required_tokens == 2065
    assert tokenizer.messages == messages
    assert tokenizer.kwargs == {
        "enable_thinking": False,
        "tools": tools,
        "documents": [{"title": "handoff", "text": "..."}],
        "tokenize": True,
        "add_generation_prompt": True,
    }


def test_budget_supports_max_tokens_alias_and_default():
    tokenizer = RecordingTokenizer(result=UserDict({"input_ids": [[1, 2, 3]]}))

    alias_budget = build_context_budget(
        tokenizer,
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64},
    )
    default_budget = build_context_budget(
        tokenizer,
        {"messages": [{"role": "user", "content": "hi"}]},
    )

    assert alias_budget.required_tokens == 67
    assert default_budget.required_tokens == 3 + DEFAULT_MAX_OUTPUT_TOKENS


def test_continue_final_message_disables_generation_prompt_by_default():
    tokenizer = RecordingTokenizer()

    build_context_budget(
        tokenizer,
        {
            "messages": [{"role": "assistant", "content": "partial"}],
            "continue_final_message": True,
        },
    )

    assert tokenizer.kwargs["continue_final_message"] is True
    assert tokenizer.kwargs["add_generation_prompt"] is False


@pytest.mark.parametrize(
    "request_data,match",
    [
        ({}, "messages must be a non-empty array"),
        ({"messages": ["invalid"]}, "each message must be a JSON object"),
        (
            {"messages": [{"role": "user"}], "max_tokens": 0},
            "must be a positive integer",
        ),
        (
            {"messages": [{"role": "user"}], "chat_template_kwargs": []},
            "must be a JSON object",
        ),
        (
            {
                "messages": [{"role": "assistant"}],
                "continue_final_message": True,
                "add_generation_prompt": True,
            },
            "cannot both be true",
        ),
    ],
)
def test_invalid_context_request_is_rejected(request_data, match):
    with pytest.raises(ContextRequestError, match=match):
        build_context_budget(RecordingTokenizer(), request_data)


def test_template_failure_is_exposed_as_request_error():
    with pytest.raises(ContextRequestError, match="unable to render chat template"):
        build_context_budget(
            RecordingTokenizer(error=ValueError("unsupported tool schema")),
            {"messages": [{"role": "user", "content": "hi"}]},
        )
