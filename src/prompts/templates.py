"""提示词模板：生成 / 有效性修复 / 充分性修复 / 自我复核。

对所有模型、所有条件保持一致，保证跨模型对比只变"模型"这一因素。
"""
from __future__ import annotations

from src.models import Problem

_SYSTEM = (
    "你是资深软件测试工程师。为给定的 Python 函数编写 pytest 单元测试。\n"
    "要求：\n"
    "1) 直接调用被测函数（函数已在环境中定义可用），不要重新定义它。\n"
    "2) 每个用例是一个以 test_ 开头的函数，使用清晰的断言（assert）。\n"
    "3) 覆盖正常、边界与异常情况，断言要能区分期望与错误实现。\n"
    "4) 用例总数控制在 6~10 个以内，优先选择最有区分度的边界，避免冗余重复；不要写参数化/循环批量断言。\n"
    "5) 只输出一个 ```python 代码块，不要额外解释文字。"
)


def _spec(problem: Problem) -> str:
    return f"【函数签名与文档】\n{problem.prompt_header}\n\n【参考实现】\n{problem.source}"


def generation_prompt(problem: Problem) -> list[dict]:
    user = (
        f"{_spec(problem)}\n\n"
        "请为上面这个函数生成一组 pytest 测试用例。"
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def validity_repair_prompt(problem: Problem, current_suite: str, error_text: str) -> list[dict]:
    user = (
        f"{_spec(problem)}\n\n"
        f"【你上一轮生成的测试】\n```python\n{current_suite}\n```\n\n"
        f"【在“正确的参考实现”上运行时的报错/失败信息】\n{error_text}\n\n"
        "参考实现本身是正确的。请修复测试，使其：语法正确、可执行、且在参考实现上全部通过"
        "（若某个断言过强导致在正确实现上失败，请修正断言或删除该用例）。只输出修复后的完整 python 代码块。"
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def adequacy_repair_prompt(problem: Problem, current_suite: str, mutant_source: str) -> list[dict]:
    user = (
        f"{_spec(problem)}\n\n"
        f"【当前通过校验的测试】\n```python\n{current_suite}\n```\n\n"
        f"【一个含有缺陷的实现变体（buggy version），但你的当前测试没能发现它】\n```python\n{mutant_source}\n```\n\n"
        "请补强/新增断言，使测试能区分这个缺陷实现（即在该变体上至少有一个用例失败），"
        "同时保证在正确的参考实现上仍然全部通过。只输出更新后的完整 python 代码块。"
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def self_review_prompt(problem: Problem, current_suite: str) -> list[dict]:
    # C3 消融：不提供任何执行/报错信息，仅让模型自我复核
    user = (
        f"{_spec(problem)}\n\n"
        f"【你上一轮生成的测试】\n```python\n{current_suite}\n```\n\n"
        "请自我审查这段测试可能存在语法或断言问题并修正它。只输出修订后的完整 python 代码块。"
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]
