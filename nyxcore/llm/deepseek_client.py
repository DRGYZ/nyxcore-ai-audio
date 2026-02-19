from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChatJSONResponse:
    data: dict[str, Any]
    usage: dict[str, Any]


def _normalize_openai_base_url(base_url: str) -> str:
    raw = base_url.strip().rstrip("/")
    if raw.lower().endswith("/v1"):
        return raw
    return raw + "/v1"


def _extract_json(text: str) -> dict[str, Any]:
    return json.loads(text.strip())


def chat_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    timeout_seconds: float = 45.0,
    max_retries: int = 3,
) -> ChatJSONResponse:
    from openai import APIConnectionError, APITimeoutError, OpenAI
    from openai import APIStatusError as OpenAIAPIStatusError
    from openai import OpenAIError

    normalized_base_url = _normalize_openai_base_url(base_url)
    client = OpenAI(api_key=api_key, base_url=normalized_base_url, timeout=timeout_seconds)

    enforce_json_once = False
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            final_user_prompt = user_prompt
            if enforce_json_once:
                final_user_prompt += "\n\nReturn ONLY JSON."

            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            if not response.choices:
                raise RuntimeError("No choices in response")
            content = response.choices[0].message.content or ""
            data = _extract_json(content)
            usage_obj = response.usage
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
            return ChatJSONResponse(data=data, usage=usage)
        except json.JSONDecodeError as exc:
            last_exc = exc
            if not enforce_json_once:
                enforce_json_once = True
                continue
        except (APIConnectionError, APITimeoutError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
        except OpenAIAPIStatusError as exc:
            last_exc = exc
            if exc.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
        except OpenAIError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        break

    raise RuntimeError(f"DeepSeek chat_json failed: {last_exc}")
