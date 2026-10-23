"""四大指标族计算：有效性 / 变异分数 / 成本(由 loop 记录) / 覆盖率。

用批量沙箱接口：一次子进程即可拿到"P 上的有效性" + "每个变异体是否被杀死"，
覆盖率再单独走一次 pytest-cov（非热路径）。
"""
from __future__ import annotations

from src.models import ExecStatus, Mutant, Problem
from src.sandbox.runner import Sandbox


def evaluate_suite(
    problem: Problem,
    suite: str,
    mutants: list[Mutant],
    sandbox: Sandbox,
) -> dict:
    empty = {
        "syntax_ok": False, "executable": False, "valid_on_p": False,
        "n_tests": 0, "n_passed": 0, "n_failed": 0,
        "total_mutants": len(mutants), "killed_mutants": 0, "mutation_score": None,
        "coverage_line": None, "coverage_branch": None,
    }
    if not suite.strip() or not sandbox.syntax_ok(suite):
        return empty

    # 一次批量：P 作为第一个变体，其余为变异体
    variants = [("P", problem.source)] + [(f"m{m.index}", m.source) for m in mutants]
    results = sandbox.exec_variants(suite, variants)

    p = results["P"]
    executable = p.status not in (ExecStatus.SYNTAX_ERROR, ExecStatus.EXEC_ERROR)
    valid_on_p = p.status == ExecStatus.VALID and p.n_tests > 0

    killed = 0
    for m in mutants:
        r = results.get(f"m{m.index}")
        if r is not None and r.killed:
            killed += 1
    ms = (killed / len(mutants)) if mutants else None

    cov_line = cov_branch = None
    if valid_on_p:  # 仅对有效套件才跑一次 pytest-cov（贵），无效则跳过以提速
        cov = sandbox.run_with_coverage(problem.source, suite)
        cov_line = cov.coverage_line
        cov_branch = cov.coverage_branch

    return {
        "syntax_ok": True,
        "executable": executable,
        "valid_on_p": valid_on_p,
        "n_tests": p.n_tests,
        "n_passed": p.n_passed,
        "n_failed": p.n_failed,
        "total_mutants": len(mutants),
        "killed_mutants": killed,
        "mutation_score": ms,
        "coverage_line": cov_line,
        "coverage_branch": cov_branch,
    }
