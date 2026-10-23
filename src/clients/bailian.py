"""阿里百炼（DashScope）OpenAI 兼容后端。三模型共用，仅 model 不同。"""
from __future__ import annotations

import time

from openai import OpenAI

from src.clients.base import LLMClient
from src.config import ModelConfig
from src.models import CompletionResult, Problem


class BailianClient(LLMClient):
    def __init__(self, cfg: ModelConfig, base_url: str, api_key: str) -> None:
        if not api_key:
            raise RuntimeError(
                "未检测到 DASHSCOPE_API_KEY。请设置环境变量后再用 --backend bailian。"
            )
        self._cfg = cfg
        self.model_key = cfg.key
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(
        self,
        messages: list[dict],
        *,
        tag: str | None = None,          # 真实后端忽略
        problem: Problem | None = None,  # 真实后端忽略
        current_suite: str | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self._cfg.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=self._cfg.max_output_tokens,
        )
        latency = int((time.perf_counter() - start) * 1000)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return CompletionResult(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency,
            model=self._cfg.model_id,
        )
