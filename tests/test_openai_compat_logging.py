from backend.server.openai_compat import chat_request_log_summary


def test_chat_request_log_summary_is_strictly_allowlisted():
    secret = "private-source-code-that-must-not-enter-worker-logs"
    summary = chat_request_log_summary(
        {
            "request_id": "request-safe-id",
            "messages": [{"role": "user", "content": secret}],
            "tools": [{"function": {"name": "read_private_file", "description": secret}}],
            "stream": True,
            "unknown_future_payload": secret,
        }
    )

    assert summary == {
        "request_id": "request-safe-id",
        "stream": True,
        "message_count": 1,
        "tool_count": 1,
    }
    assert secret not in repr(summary)
    assert "read_private_file" not in repr(summary)
