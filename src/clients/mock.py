"""Mock 后端：不消耗任何 API 额度，用于离线打通端到端流水线。

行为设计（刻意用来演示闭环 mechanics）：
- GEN：生成一个"断言过强 / 首个用例在正确实现上失败"的套件 -> 触发有效性修复
- REPAIR_VALIDITY：返回正确套件 -> 变为 VALID
- ADEQUACY：原样返回当前套件（Mock 不真正推理变异体，仅验证流程）
- SELF_REVIEW：原样返回当前套件（无执行反馈时无法收敛，体现 C3 的劣势）
"""
from __future__ import annotations

import time

from src.clients.base import LLMClient
from src.models import CompletionResult, Problem
from src.utils import estimate_tokens


def _wrong(expected: str) -> str:
    e = expected.strip()
    if e == "True":
        return "False"
    if e == "False":
        return "True"
    try:
        return str(int(e) + 1000)
    except ValueError:
        return repr("__MOCK_WRONG__")


def _suite(problem: Problem, *, mutate_first: bool = False) -> str:
    lines: list[str] = []
    for i, ex in enumerate(problem.examples):
        exp = _wrong(ex.expected) if (mutate_first and i == 0) else ex.expected
        lines.append(f"def test_ex{i}_{problem.entry_point}():")
        lines.append(f"    assert {problem.entry_point}({ex.args}) == {exp}")
        lines.append("")
    body = "\n".join(lines) or f"def test_{problem.entry_point}_none():\n    assert True\n"
    return "```python\n" + body + "\n```"


class MockClient(LLMClient):
    def __init__(self, model_key: str, display: str) -> None:
        self.model_key = model_key
        self._display = display

    def complete(
        self,
        messages: list[dict],
        *,
        tag: str | None = None,
        problem: Problem | None = None,
        current_suite: str | None = None,
        temperature: float = 0.0,
    ) -> CompletionResult:
        start = time.perf_counter()
        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        if problem is None:
            text = "```python\ndef test_placeholder():\n    assert True\n```"
        elif tag == "GEN":
            text = _suite(problem, mutate_first=True)
        elif tag == "REPAIR_VALIDITY":
            text = _suite(problem, mutate_first=False)
        elif tag in ("ADEQUACY", "SELF_REVIEW"):
            text = "```python\n" + (current_suite or _suite(problem, mutate_first=False)) + "\n```"
        else:
            text = _suite(problem, mutate_first=True)

        latency = int((time.perf_counter() - start) * 1000)
        return CompletionResult(
            text=text,
            prompt_tokens=max(1, prompt_chars // 4),
            completion_tokens=estimate_tokens(text),
            latency_ms=max(1, latency),
            model=f"mock:{self._display}",
        )
