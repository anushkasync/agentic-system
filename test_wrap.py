
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
