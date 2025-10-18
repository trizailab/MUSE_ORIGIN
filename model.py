import os
import yaml
import httpx
import base64
import aiofiles
import traceback
import json
from pathlib import Path
from datetime import datetime
from threading import Lock
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import AsyncGenerator, Union

load_dotenv()

with open("config.yaml", "r") as f:
    raw_config = os.path.expandvars(f.read())
    config = yaml.safe_load(raw_config)
LLM_CONFIG = config["llm"]


class TokenLogger:
    """Logger for recording token usage and costs to TOKENS-LOG.md"""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.log_file = Path("TOKENS-LOG.md")

            # Create file with header if it doesn't exist
            if not self.log_file.exists():
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write("# MUSE Token Usage Log\n\n")
                    f.write("Each line below is a JSON object representing one LLM request:\n\n")

            self.initialized = True

    def log_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        input_price: float,
        output_price: float
    ):
        """Log a single LLM request with token usage and cost"""
        try:
            # Calculate cost (prices are per 1 million tokens)
            total_cost = (
                (input_tokens * input_price / 1_000_000) +
                (output_tokens * output_price / 1_000_000)
            )

            # Format date
            date_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

            # Create JSON entry
            log_entry = {
                "date": date_str,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": round(total_cost, 6)
            }

            # Write to file (append, new line)
            with self._lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        except Exception as e:
            # Don't fail if logging doesn't work
            print(f"⚠️ Warning: Failed to log tokens: {e}")

class LLM:
    NUM_CALLS = 0
    PROMPT_TOKENS = 0
    COMPLETION_TOKENS = 0
    MAX_TOKENS = 0

    def __init__(self, model: str="Qwen2.5-VL-7B-Instruct"):
        cfg = LLM_CONFIG.get(model)
        if cfg is None:
            raise ValueError(f"Model '{model}' not found in config.yaml")
        self.async_client = AsyncOpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            http_client=httpx.AsyncClient(verify=False),
            timeout=180
        )
        self.model = cfg["model"]
        self.model_name = model  # Store model name from config for logging

        # Token prices (defaults to 0.0 if not specified)
        self.input_price = cfg.get("input_tokens_price", 0.0)
        self.output_price = cfg.get("output_tokens_price", 0.0)

        # Initialize token logger
        self.token_logger = TokenLogger()

    def _accumulate_usage(self, usage):
        """Accumulate usage statistics and log to file"""
        get = (lambda k, default=0:
               usage.get(k, default) if isinstance(usage, dict)
               else getattr(usage, k, default))
        prompt_tokens = int(get("prompt_tokens", 0) or 0)
        completion_tokens = int(get("completion_tokens", 0) or 0)

        # Accumulate static counters (as before)
        LLM.PROMPT_TOKENS += prompt_tokens
        LLM.COMPLETION_TOKENS += completion_tokens
        LLM.MAX_TOKENS = max(
            LLM.MAX_TOKENS,
            prompt_tokens + completion_tokens
        )

        # Log this specific request
        if prompt_tokens > 0 or completion_tokens > 0:
            self.token_logger.log_request(
                model=self.model_name,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                input_price=self.input_price,
                output_price=self.output_price
            )

    async def async_generate(
            self,
            prompt: str,
            image_path: Union[str, Path, None] = None,
            history: list[dict] = None,
            max_tokens: Union[int, None] = 32768
    ) -> str:
        LLM.NUM_CALLS += 1
        try:
            messages = await self.prepare_messages(prompt, image_path, history)

            resp = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )

            usage = getattr(resp, "usage", None)
            if usage:
                self._accumulate_usage(usage)

            choices = getattr(resp, "choices", None) or []
            if not choices:
                print("[SYSTEM WARNING][SYNC] ⚠️ No choices in response.")
                print(resp)
                return self._handle_error(RuntimeError("Empty choices from LLM response."))

            c0 = choices[0]

            self._log_finish_reason("SYNC", getattr(c0, "finish_reason", None))

            msg = getattr(c0, "message", None)

            content = getattr(msg, "content", None) if msg else None
            if content is None:
                print("[SYSTEM WARNING][SYNC] ⚠️ Response has no content (may contain only tool/function signals).")
                return self._handle_error(RuntimeError("Empty content in first choice."))

            return content

        except Exception as e:
            return self._handle_error(e)

    async def async_stream_generate(
            self,
            prompt: str,
            image_path: Union[str, Path, None] = None,
            history: list[dict] = None,
            max_tokens: Union[int, None] = 32768,
            temperature: float = 1.0
    ) -> AsyncGenerator[str, None]:
        LLM.NUM_CALLS += 1
        try:
            messages = await self.prepare_messages(prompt, image_path, history)

            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                stream_options={"include_usage": True}
            )

            saw_explicit_finish = False
            usage_accumulated = False

            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage and not usage_accumulated:
                    self._accumulate_usage(usage)
                    usage_accumulated = True

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                c0 = choices[0]

                finish_reason = getattr(c0, "finish_reason", None)
                if finish_reason is not None:
                    saw_explicit_finish = True
                    self._log_finish_reason("STREAM", finish_reason)

                delta = getattr(c0, "delta", None)

                if not usage_accumulated and delta is not None:
                    maybe_usage = getattr(delta, "usage", None)
                    if maybe_usage:
                        self._accumulate_usage(maybe_usage)
                        usage_accumulated = True

                content = getattr(delta, "content", None) if delta else None
                if content is not None:
                    yield content

            if not saw_explicit_finish:
                print("[SYSTEM INFO][STREAM] ℹ️ Stream ended without explicit finish_reason (likely normal).")

        except Exception as e:
            yield self._handle_error(e)

    async def prepare_messages(
        self,
        prompt: str,
        image_path: Union[str, Path, None],
        history: list[dict] = None
    ) -> list[dict]:
        messages = history.copy() if history else []

        if image_path:
            base64_image = await self.image_to_base64(image_path)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        else:
            content = [{"type": "text", "text": prompt}]

        messages.append(
            {"role": "user", "content": content}
        )
        return messages

    def _handle_error(self, e: Exception) -> str:
        print(f"==========Error: {e}==========")
        print(traceback.format_exc())
        print(f"==========Model: {self.model}==========")
        return f"ERROR: {type(e).__name__} - {str(e)}"

    @staticmethod
    def _log_finish_reason(where: str, finish_reason: str | None):
        if finish_reason is None:
            return
        # if finish_reason == "stop":
        #     print(f"[SYSTEM INFO][{where}] ✅ finish_reason=stop (normal completion)")
        # elif finish_reason == "length":
        #     print(f"[SYSTEM WARNING][{where}] ⚠️ finish_reason=length (max_tokens reached, text truncated)")
        # elif finish_reason == "content_filter":
        #     print(f"[SYSTEM WARNING][{where}] ⚠️ finish_reason=content_filter (content security/compliance filtering hit)")
        # elif finish_reason == "tool_calls":
        #     print(f"[SYSTEM WARNING][{where}] ⚠️ finish_reason=tool_calls (model suggests calling a tool, may return a tool_calls structure)")
        # elif finish_reason == "function_call":
        #     print(f"[SYSTEM WARNING][{where}] ⚠️ finish_reason=function_call (model suggests function call, legacy/compatible fields)")
        # else:
        #     print(f"[SYSTEM WARNING][{where}] ⚠️ Unknown finish_reason={finish_reason} (Unknown/Vendor Custom Extension)")

    @staticmethod
    async def image_to_base64(image_path: Union[str, Path]) -> str:
        async with aiofiles.open(image_path, "rb") as image_file:
            content = await image_file.read()
            encoded_string = base64.b64encode(content).decode("utf-8")
        return encoded_string

if __name__ == "__main__":
    print(LLM_CONFIG)

    import asyncio

    async def test():
        llm = LLM(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

        history = [
            {"role": "user", "content": [{"type": "text", "text": "You are Long Aotian from Class 3-1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Yes, I am Long Aotian from Class 3-1."}]}
        ]

        async for chunk in llm.async_stream_generate("Hello, please introduce yourself.", history=history):
            print(chunk, end="")

        print("\n[USAGE] prompt =", LLM.PROMPT_TOKENS, "completion =", LLM.COMPLETION_TOKENS)

    asyncio.run(test())
