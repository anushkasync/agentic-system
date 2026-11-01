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
    max_output_tokens = 256
    max_output_tokens_cap = 1024

    start = time.monotonic()

    max_tokens = max_output_tokens

    for attempt in range(max_attempts):
        
        try:
            messages = [{"role": "user", "content":"List exactly 3 colors"}]

            content_parts = []
            max_continuations = 2
            completed = False

            for i in range(max_continuations+1):
                remaining = total_time_allowed - (
                    time.monotonic() - start
                )
                if remaining <= 0:
                    raise TimeoutError("Total time budget exceeded")
                
                request_timeout = httpx.Timeout(
                    timeout=min(10.0, remaining),
                    connect=min(2.0, remaining),
                    write=min(3.0, remaining),
                    read=min(10.0, remaining),
                    pool=min(0.5, remaining),
                )

                request_args = {
                    "model": "openai/gpt-oss-20b",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "timeout": request_timeout,
                }

                if i == 0:
                    request_args["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Response",
                            "schema": Response.model_json_schema(),
                            "strict": True,
                        },
                    }

                completion = client.chat.completions.create(
                    **request_args
                )

                choice = completion.choices[0]

                if choice.finish_reason == "content_filter":
                    raise ValueError(
                        "LLM Failure: Output blocked by content filter."
                    )

                partial_content = choice.message.content or ""

                # Save the generated text
                content_parts.append(partial_content)

                # Generation completed successfully
                if choice.finish_reason != "length":
                    completed = True
                    break

                # No continuation attempts remaining
                if i == max_continuations:
                    break

                # Preserve the partial assistant response
                messages.append({
                    "role": "assistant",
                    "content": partial_content,
                })

                # Ask the model to continue
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was cut off. "
                        "Continue exactly where it ended. "
                        "Output only the remaining JSON text. "
                        "Do not repeat the existing content."
                    ),
                })

            if not completed:
                raise ValueError(
                    "LLM Failure: Response remained truncated "
                    "after continuation attempts."
                )

            # Combine all generated fragments
            content = "".join(content_parts)

            # Validate the reconstructed JSON
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
