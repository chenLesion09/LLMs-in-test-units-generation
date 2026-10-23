"""LLM 客户端抽象层。

真实后端只需实现 complete()；Mock 后端用于离线打通流水线。
所有后端共享同一 messages 结构，保证"跨模型只变 model 一个因素"。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import CompletionResult, Problem


class LLMClient(ABC):
    """统一补全接口。

    tag / problem / current_suite 仅供 Mock 后端模拟闭环行为；
    真实后端忽略这些参数，仅依据 messages 调用 API。
    """

    model_key: str

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        tag: str | None = None,
        problem: Problem | None = None,
        current_suite: str | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        ...
