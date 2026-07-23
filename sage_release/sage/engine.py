"""vLLM completions client and engine wrappers."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import aiohttp
import requests


DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B-FP8"


class VLLMCompletionsClient:
    """Thin HTTP client for a vLLM /v1/completions endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:9090/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        model: str = DEFAULT_MODEL_NAME,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "")
        self.timeout = timeout
        self.session = requests.Session()
        self.model = model

    def completions(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **extra,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if logprobs is not None:
            payload["logprobs"] = int(logprobs)
        if stop:
            payload["stop"] = stop
        if extra:
            payload.update(extra)

        resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    async def acompletions(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **extra,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if logprobs is not None:
            payload["logprobs"] = int(logprobs)
        if stop:
            payload["stop"] = stop
        if extra:
            payload.update(extra)

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()


class AugEngine:
    """Engine wrapper that runs Qwen3 in non-thinking mode.

    The prompt is wrapped in the Qwen3 chat template with the assistant turn
    prefilled with an empty ``<think></think>`` block. This is how Qwen3
    non-thinking mode is actually engaged. Appending a bare ``/nothink`` to a
    raw /v1/completions prompt does NOT engage it: the switch only takes effect
    through the chat template, so the model would otherwise emit a full chain of
    thought that gets truncated at max_tokens before it reaches \\boxed{}.
    The completions endpoint (not chat) is kept so token logprobs remain
    available for SAGE contrastive scoring.
    """

    _USER_TEMPLATE = (
        "<|im_start|>user\n{content}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    def __init__(self, client: VLLMCompletionsClient):
        self.client = client
        self.verbose = False

    @classmethod
    def _prepare(cls, prompt: str, kwargs: dict) -> str:
        """Wrap prompt in the non-thinking template and add the turn stop token."""
        stop = kwargs.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        stop = list(stop)
        if "<|im_end|>" not in stop:
            stop.append("<|im_end|>")
        kwargs["stop"] = stop
        return cls._USER_TEMPLATE.format(content=prompt)

    def __call__(
        self,
        prompt: str | list,
        n: int = 1,
        logprobs: int = 0,
        return_tokens: bool = False,
        **kwargs,
    ) -> str:
        prompt = self._prepare(prompt, kwargs)
        if self.verbose:
            print("--> Generating:")
            if "system_prompt" in kwargs:
                print(f"System prompt: {kwargs['system_prompt']}")
            print(f"Prompt: {prompt}\n...")
        ret = self.client.completions(prompt, n=n, logprobs=logprobs, **kwargs)
        texts, all_logprobs, all_tokens = [], [], []
        for i, choice in enumerate(ret["choices"]):
            texts.append(choice["text"].replace("<think>\n\n</think>\n\n", ""))
            all_logprobs.append(choice["logprobs"]["top_logprobs"])
            all_tokens.append(choice["logprobs"]["tokens"])
            if self.verbose:
                print(f"--> Generated {i + 1}/{n}\n{texts[-1]}\n")

        if logprobs > 0:
            if return_tokens:
                return texts, all_logprobs, all_tokens
            return texts, all_logprobs
        if return_tokens:
            return texts, all_tokens
        return texts

    def generate(self, prompt: str, n: int = 1, **kwargs) -> str:
        return self.__call__(prompt, n, **kwargs)

    async def agenerate(
        self,
        prompt: str | list,
        n: int = 1,
        logprobs: int = 0,
        return_tokens: bool = False,
        **kwargs,
    ) -> str:
        prompt = self._prepare(prompt, kwargs)
        if self.verbose:
            print("--> Generating:")
            if "system_prompt" in kwargs:
                print(f"System prompt: {kwargs['system_prompt']}")
            print(f"Prompt: {prompt}\n...")
        ret = await self.client.acompletions(prompt, n=n, logprobs=logprobs, **kwargs)
        texts, all_logprobs, all_tokens = [], [], []
        for i, choice in enumerate(ret["choices"]):
            texts.append(choice["text"].replace("<think>\n\n</think>\n\n", ""))
            all_logprobs.append(choice["logprobs"]["top_logprobs"])
            all_tokens.append(choice["logprobs"]["tokens"])
            if self.verbose:
                print(f"--> Generated {i + 1}/{n}\n{texts[-1]}\n")

        if logprobs > 0:
            if return_tokens:
                return texts, all_logprobs, all_tokens
            return texts, all_logprobs
        if return_tokens:
            return texts, all_tokens
        return texts


class EngineNoThink:
    """Engine wrapper without any prompt modification (raw completions)."""

    def __init__(self, client: VLLMCompletionsClient):
        self.client = client
        self.verbose = False

    def __call__(
        self,
        prompt: str | list,
        n: int = 1,
        logprobs: int = 0,
        return_tokens: bool = False,
        **kwargs,
    ) -> str:
        ret = self.client.completions(prompt, n=n, logprobs=logprobs, **kwargs)
        texts, all_logprobs, all_tokens = [], [], []
        for i, choice in enumerate(ret["choices"]):
            texts.append(choice["text"])
            all_logprobs.append(choice["logprobs"]["top_logprobs"])
            all_tokens.append(choice["logprobs"]["tokens"])
            if self.verbose:
                print(f"--> Generated {i + 1}/{n}\n{texts[-1]}\n")

        if logprobs > 0:
            if return_tokens:
                return texts, all_logprobs, all_tokens
            return texts, all_logprobs
        if return_tokens:
            return texts, all_tokens
        return texts

    def generate(self, prompt: str, n: int = 1, **kwargs) -> str:
        return self.__call__(prompt, n, **kwargs)

    async def agenerate(
        self,
        prompt: str | list,
        n: int = 1,
        logprobs: int = 0,
        return_tokens: bool = False,
        **kwargs,
    ) -> str:
        ret = await self.client.acompletions(prompt, n=n, logprobs=logprobs, **kwargs)
        texts, all_logprobs, all_tokens = [], [], []
        for i, choice in enumerate(ret["choices"]):
            texts.append(choice["text"])
            all_logprobs.append(choice["logprobs"]["top_logprobs"])
            all_tokens.append(choice["logprobs"]["tokens"])
            if self.verbose:
                print(f"--> Generated {i + 1}/{n}\n{texts[-1]}\n")

        if logprobs > 0:
            if return_tokens:
                return texts, all_logprobs, all_tokens
            return texts, all_logprobs
        if return_tokens:
            return texts, all_tokens
        return texts


class ThinkEngine(AugEngine):
    """Qwen3 THINKING mode: chat template WITHOUT the prefilled empty think block,
    so the model emits its own <think>...</think> before the answer. Same completions
    endpoint and logprobs handling as AugEngine (only the assistant-turn prefix differs).
    Callers should pass a large max_tokens (thinking is long)."""

    _USER_TEMPLATE = (
        "<|im_start|>user\n{content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_engine(
    base_url: str = "http://localhost:9090/v1",
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    model: str = DEFAULT_MODEL_NAME,
    type: str = "aug",
) -> AugEngine:
    """Factory for engine instances.

    type:
      "aug"      — AugEngine (Qwen3 non-thinking via chat template + empty think block)
      "think"    — ThinkEngine (Qwen3 thinking mode; model generates its own reasoning)
      "no_think" — EngineNoThink (raw completions, no prompt modification)
    """
    client = VLLMCompletionsClient(base_url=base_url, api_key=api_key, timeout=timeout, model=model)
    if type == "aug":
        return AugEngine(client)
    elif type == "think":
        return ThinkEngine(client)
    elif type == "no_think":
        return EngineNoThink(client)
    else:
        raise ValueError(f"Invalid engine type: {type!r}. Choose 'aug', 'think' or 'no_think'.")
