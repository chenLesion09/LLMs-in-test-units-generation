"""构造"有效变异体集"：只保留能被官方测试杀死的变异体（即非等价体）。

对齐设计文档 §5.3：官方测试杀得掉的，说明它确实改变了被规约的行为，
用它同时充当阶段二的反例池与变异分数的分母。
"""
from __future__ import annotations

from src.models import Mutant, Problem
from src.mutation.operators import enumerate_mutants
from src.sandbox.runner import Sandbox


def build_non_equivalent_mutants(problem: Problem, sandbox: Sandbox) -> list[Mutant]:
    if not problem.reference_tests.strip():
        # 没有官方测试就无法过滤等价体；保守返回空集（该题不计入变异分数）
        return []

    kept: list[Mutant] = []
    # 一次批量：所有候选变异体在同一进程内评估，官方测试杀得掉的即为非等价体
    cands = [m for m in enumerate_mutants(problem.source) if m.source.strip() != problem.source.strip()]
    results = sandbox.exec_variants(problem.reference_tests,
                                    [(f"m{m.index}", m.source) for m in cands])
    for m in cands:
        r = results.get(f"m{m.index}")
        # 官方测试在该变异体上失败/报错 => 变异体被杀死 => 非等价，保留
        if r is not None and r.killed:
            kept.append(m)

    for i, m in enumerate(kept):
        m.index = i
    return kept
