"""闭环控制器：实现 C0-C3 四种条件的数据流（见设计文档 §3 / §3.1）。"""
from __future__ import annotations

import ast

from src.clients.base import LLMClient
from src.models import Condition, ExecStatus, Mutant, Problem, RoundRecord, TrialResult
from src.prompts import templates
from src.sandbox.runner import Sandbox
from src.utils import extract_code, truncate_traceback


class _Loop:
    def __init__(
        self, client: LLMClient, sandbox: Sandbox, trial: TrialResult,
        budget, window: int, temperature: float,
    ) -> None:
        self.client = client
        self.sandbox = sandbox
        self.trial = trial
        self.budget = budget
        self.window = window
        self.temp = temperature
        self.final_suite = ""
        self.final_status: ExecStatus = ExecStatus.EMPTY
        self._pt = 0
        self._ct = 0

    # ---- 成本记账 + 调用 ----
    def can_call(self) -> bool:
        return self.trial.llm_calls < self.budget.max_llm_calls

    def _chat(self, messages, tag, problem, current_suite=None) -> str:
        comp = self.client.complete(
            messages, tag=tag, problem=problem, current_suite=current_suite, temperature=self.temp
        )
        self._pt, self._ct = comp.prompt_tokens, comp.completion_tokens
        t = self.trial
        t.llm_calls += 1
        t.total_prompt_tokens += comp.prompt_tokens
        t.total_completion_tokens += comp.completion_tokens
        t.peak_prompt_tokens = max(t.peak_prompt_tokens, comp.prompt_tokens)
        if self.window and comp.prompt_tokens / self.window > self.budget.window_hit_ratio:
            t.window_hit_events += 1
        return extract_code(comp.text)

    def _record(self, kind: str, status: ExecStatus) -> None:
        self.trial.rounds.append(RoundRecord(
            index=len(self.trial.rounds), kind=kind, status=status.value,
            prompt_tokens=self._pt, completion_tokens=self._ct,
        ))

    def _run(self, problem: Problem, suite: str) -> ExecStatus:
        if not suite.strip():
            return ExecStatus.EMPTY
        return self.sandbox.exec_single(problem.source, suite).status

    def _traceback(self, suite: str) -> str:
        if not self.sandbox.syntax_ok(suite):
            return "SyntaxError: 生成的测试无法解析（语法错误）。"
        res = self.sandbox.exec_single(self.problem_source, suite)
        return truncate_traceback(res.error_text)

    # ---- 条件实现 ----
    def c0(self, problem: Problem) -> None:
        suite = self._chat(templates.generation_prompt(problem), "GEN", problem)
        st = self._run(problem, suite)
        self._record("gen", st)
        self.final_suite, self.final_status = suite, st

    def c1(self, problem: Problem) -> None:
        suite = self._chat(templates.generation_prompt(problem), "GEN", problem)
        st = self._run(problem, suite)
        pruned = _prune_invalid_tests(problem, suite, self.sandbox) if st == ExecStatus.FAIL_ON_P else suite
        final_st = self._run(problem, pruned)
        self._record("gen", final_st)
        self.final_suite, self.final_status = pruned, final_st

    def c3(self, problem: Problem) -> None:
        suite = self._chat(templates.generation_prompt(problem), "GEN", problem)
        st = self._run(problem, suite)
        self._record("gen", st)
        # 自我复核循环：回路里没有执行信息
        for _ in range(self.budget.max_rounds - 1):
            if not self.can_call():
                break
            suite = self._chat(templates.self_review_prompt(problem, suite), "SELF_REVIEW", problem, suite)
            st = self._run(problem, suite)  # 仅用于记录，不作为反馈喂回
            self._record("self_review", st)
        self.final_suite, self.final_status = suite, st

    def c2(self, problem: Problem, mutants: list[Mutant]) -> None:
        self.problem_source = problem.source
        suite = self._chat(templates.generation_prompt(problem), "GEN", problem)
        st = self._run(problem, suite)
        self._record("gen", st)

        # 阶段一：有效性闭环
        rounds = 1
        while st != ExecStatus.VALID and rounds < self.budget.max_rounds and self.can_call():
            err = self._traceback(suite)
            suite = self._chat(
                templates.validity_repair_prompt(problem, suite, err), "REPAIR_VALIDITY", problem, suite
            )
            st = self._run(problem, suite)
            self._record("repair_validity", st)
            rounds += 1

        if st != ExecStatus.VALID:
            self.final_suite, self.final_status = suite, st
            return

        # 阶段二：变异引导的充分性修复（轮次单独收紧，默认 max_adequacy_rounds）
        survivors = _survivors(problem, suite, mutants, self.sandbox)
        adequacy_cap = getattr(self.budget, "max_adequacy_rounds", self.budget.max_rounds)
        adequacy_rounds = 0
        while survivors and adequacy_rounds < adequacy_cap and self.can_call():
            target = survivors[0]
            new_suite = self._chat(
                templates.adequacy_repair_prompt(problem, suite, target.source), "ADEQUACY", problem, suite
            )
            st2 = self._run(problem, new_suite)  # 复检有效性
            if st2 == ExecStatus.VALID:
                new_survivors = _survivors(problem, new_suite, mutants, self.sandbox)
                if len(new_survivors) < len(survivors):  # 只有真正多杀了才接受
                    suite, survivors = new_suite, new_survivors
                    self._record("adequacy", ExecStatus.VALID)
                else:
                    self._record("adequacy", st2)
            else:
                self._record("adequacy", st2)  # 回退，保留旧 suite
            adequacy_rounds += 1

        self.final_suite, self.final_status = suite, ExecStatus.VALID


def _survivors(problem: Problem, suite: str, mutants: list[Mutant], sandbox: Sandbox) -> list[Mutant]:
    if not mutants:
        return []
    variants = [(f"m{m.index}", m.source) for m in mutants]
    results = sandbox.exec_variants(suite, variants)
    return [m for m in mutants if not results.get(f"m{m.index}").killed]


def _prune_invalid_tests(problem: Problem, suite: str, sandbox: Sandbox) -> str:
    """C1：一次执行拿到逐用例结果，丢弃在参考实现上失败的用例，不回喂模型。"""
    try:
        tree = ast.parse(suite)
    except SyntaxError:
        return ""
    res = sandbox.exec_single(problem.source, suite)
    passing = {name for name, st in res.tests if st == "pass"}
    kept: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and node.name in passing:
            seg = ast.get_source_segment(suite, node) or ""
            if seg:
                kept.append(seg)
    return "\n\n".join(kept)


def run_trial(
    problem: Problem, condition: Condition, client: LLMClient, sandbox: Sandbox,
    budget, window: int, temperature: float, mutants: list[Mutant],
) -> TrialResult:
    trial = TrialResult(model=client.model_key, condition=condition.value, task_id=problem.task_id)
    loop = _Loop(client, sandbox, trial, budget, window, temperature)
    if condition == Condition.C0:
        loop.c0(problem)
    elif condition == Condition.C1:
        loop.c1(problem)
    elif condition == Condition.C3:
        loop.c3(problem)
    elif condition == Condition.C2:
        loop.c2(problem, mutants)
    trial.final_suite = loop.final_suite
    trial.final_status = loop.final_status.value
    return trial
