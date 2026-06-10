"""Portable chat-message helpers.

Vendored verbatim from ``mlx_lm.server`` (Apple, MIT License) so the shared /
vLLM code path does not depend on ``mlx_lm`` — mlx is Apple-Silicon/Linux only
and absent on Windows, where Parallax now contributes via the native vLLM
backend. Behaviour is identical to the upstream functions.

Source: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py
"""

import json
from typing import List, Optional


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "stop": "\n",
    }
    role_mapping = role_mapping or default_role_mapping

    prompt = ""
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")
        stop = role_mapping.get("stop", "")
        content = line.get("content", "")
        prompt += f"{role_prefix}{content}{stop}"

    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


def process_message_content(messages):
    """
    Convert message content to a format suitable for `apply_chat_template`.

    The function operates on messages in place. Content of type "text" is
    flattened to a plain string; ``None`` content becomes an empty string; and
    tool-call argument strings are parsed from JSON.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [
                fragment["text"] for fragment in content if fragment["type"] == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""

        if tool_calls := message.get("tool_calls"):
            for tool_call in tool_calls:
                if func := tool_call.get("function"):
                    if args := func.get("arguments"):
                        func["arguments"] = json.loads(args)
