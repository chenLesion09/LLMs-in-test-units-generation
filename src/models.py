"""领域数据模型：问题、模型补全、变异体、执行状态、轮次与试次记录。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


@dataclass
class Example:
    """一个输入->期望输出示例，用于 Mock 构造测试，也可为 prompt 提供样例。"""

    args: str        # 调用参数表达式，如 "3, 1, 2"
    expected: str    # 期望结果的 repr，如 "True"


@dataclass
class Problem:
    """一道被测程序题（函数级），对齐 HumanEval / MBPP 的结构。"""

    task_id: str
    entry_point: str         # 函数名
    prompt_header: str       # 函数签名 + docstring（喂给模型的规格）
    source: str              # 完整正确实现 P（含函数定义）
    examples: list[Example] = field(default_factory=list)
    reference_tests: str = ""  # 官方测试（用于变异体等价过滤 + 覆盖基线）

    def to_ref(self) -> dict:
        return {"task_id": self.task_id, "entry_point": self.entry_point}


@dataclass
class CompletionResult:
    """一次 LLM 调用的返回。"""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Mutant:
    index: int
    source: str          # 变异后的完整函数源码
    description: str     # 人类可读：改了哪个算子/位置


class ExecStatus(str, Enum):
    SYNTAX_ERROR = "syntax_error"   # 无法 ast.parse
    EXEC_ERROR = "exec_error"       # 收集/导入/运行期异常
    FAIL_ON_P = "fail_on_p"         # 在正确实现 P 上断言失败（oracle 过强）
    VALID = "valid"                 # 可执行且在 P 上全绿
    EMPTY = "empty"                 # 未收集到任何测试


class Condition(str, Enum):
    C0 = "C0"  # 零样本一次性生成
    C1 = "C1"  # 只校验不修复（过滤）
    C2 = "C2"  # 完整闭环
    C3 = "C3"  # 自我复核（无执行反馈，消融）


@dataclass
class RoundRecord:
    index: int
    kind: str                        # gen | repair_validity | adequacy | self_review
    status: str                      # ExecStatus 值
    prompt_tokens: int
    completion_tokens: int
    killed_before: int | None = None
    killed_after: int | None = None


@dataclass
class TrialResult:
    """单个 (模型 × 条件 × 题目) 试次的完整记录，落 JSONL。"""

    model: str
    condition: str
    task_id: str
    rounds: list[RoundRecord] = field(default_factory=list)
    llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    peak_prompt_tokens: int = 0
    window_hit_events: int = 0
    final_status: str = ""
    metrics: dict = field(default_factory=dict)
    final_suite: str = ""

    def to_log(self) -> dict:
        d = asdict(self)
        d["rounds"] = [asdict(r) for r in self.rounds]
        d["total_tokens"] = self.total_prompt_tokens + self.total_completion_tokens
        return d
