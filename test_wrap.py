import json
import httpx
import pytest
import respx
import openai
import core


@respx.mock(assert_all_mocked=True)
def test_server_processing_lag():
    route = respx.post(
        "https://api.groq.com/openai/v1/chat/completions"
    ).mock(
        side_effect=httpx.ReadTimeout("Simulated read timeout")
    )

    # Force the test client to send requests through RESpx.
    test_http = httpx.Client(
        transport=httpx.MockTransport(respx.mock.handler)
    )

    test_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        http_client=test_http,
        max_retries = 0
    )

    # Replace the production client only for this test.
    core.client = test_client

    try:
        with pytest.raises(openai.APITimeoutError):
            core.wrapper(total_time_allowed=5, max_attempts=1)

        assert route.called
        assert respx.calls.call_count == 1
        print("Passed: timeout was simulated; no real API call.")
    finally:
        test_http.close()


def mock_payload(content, finish_reason):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "openai/gpt-oss-20b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": finish_reason,
        }],
    }


@respx.mock
def test_token_truncation_continuation_flow(monkeypatch):

    url = "https://api.groq.com/openai/v1/chat/completions"

    route = respx.post(url).mock(
        side_effect=[
            httpx.Response(
                200,
                json=mock_payload(
                    '{"colors": ["crimson", "emerald"',
                    "length"
                )
            ),
            httpx.Response(
                200,
                json=mock_payload(
                    ', "sapphire"]}',
                    "stop"
                )
            ),
        ]
    )

    # Create a test client connected to the RESpx mock handler.
    test_http = httpx.Client(
        transport=httpx.MockTransport(respx.mock.handler)
    )

    test_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        http_client=test_http,
        max_retries=0,
    )

    monkeypatch.setattr(core, "client", test_client)

    try:
        result = core.wrapper(
            total_time_allowed=10,
            max_attempts=1
        )

        # 1. Verify final result.
        assert result.colors == [
            "crimson", "emerald", "sapphire"
        ]

        # 2. Inspect the second request.
        second_request = json.loads(
            respx.calls[1].request.content
        )

        messages = second_request["messages"]

        # 3. Verify previous assistant response was preserved.
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == (
            '{"colors": ["crimson", "emerald"'
        )

        # 4. Verify continuation instruction was added.
        assert messages[2]["role"] == "user"
        assert "Continue exactly where it ended" in (
            messages[2]["content"]
        )

        print("Passed: truncation, state preservation and continuation!")

    finally:
        test_http.close()