from pydantic import ConfigDict
import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import time
import random
import httpx

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
    timeout=httpx.Timeout(
        10.0,
        connect=2.0,
        write=3.0,
        read=10.0,
        pool=0.5,
    ),
)

class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    colors: list[str]

def wrapper(total_time_allowed=20, max_attempts=3):
    start = time.monotonic()

    for attempt in range(max_attempts):
        remaining = total_time_allowed - (time.monotonic() - start)

        if remaining <= 0:
            print("OVERALL BUDGET EXPIRED BEFORE REQUEST")
            raise TimeoutError("Total time budget exceeded")

        request_timeout = httpx.Timeout(
            timeout=min(10.0, remaining),
            connect=min(2.0, remaining),
            write=min(3.0, remaining),
            read=min(10.0, remaining),
            pool=min(0.5, remaining),
        )

        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "user", "content": "List exactly 3 colors."}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Response",
                        "schema": Response.model_json_schema(),
                        "strict": True,
                    },
                },
                timeout=request_timeout,
            )

            content = completion.choices[0].message.content
            result = Response.model_validate_json(content)
            print(result.model_dump_json())
            return result

        except openai.RateLimitError as e:
            if attempt == max_attempts - 1:
                raise

            retry_after = (
                e.response.headers.get("retry-after")
                if e.response is not None else None
            )

            if retry_after is not None:
                delay = float(retry_after)
            else:
                delay = min(2 ** attempt, 8) + random.uniform(0, 1)

            remaining = total_time_allowed - (time.monotonic() - start)

            if delay >= remaining:
                raise TimeoutError("Not enough time left to retry") from e

            time.sleep(delay)

    raise RuntimeError("Request attempts exhausted")

if __name__ == "__main__":
    wrapper()
