"""数据集加载器。

默认用离线 samples 迷你基准；提供 HumanEval / MBPP 的真实加载入口（需可选依赖 datasets）。
"""
from __future__ import annotations

from src.dataset.samples import get_sample_problems
from src.models import Example, Problem


def _humaneval_to_problem(row: dict) -> Problem:
    """HumanEval 行 -> Problem。

    官方 test 形如 `def check(candidate): ...`，这里包装成 pytest 用例调用真实函数名，
    以便在 import * 的沙箱里与生成套件同构执行。
    """
    entry = row["entry_point"]
    test = row.get("test", "") or ""
    ref = test + f"\n\ndef test_reference_check():\n    check({entry})\n"
    return Problem(
        task_id=f"he_{row['task_id']}",
        entry_point=entry,
        prompt_header=row["prompt"],
        source=row["prompt"] + row.get("canonical_solution", ""),
        examples=[],
        reference_tests=ref,
    )


def _mbpp_to_problem(row: dict) -> Problem:
    """MBPP 行 -> Problem。

    full 配置的真实列为：task_id / text / code / test_list / test_setup_code / challenge_test_list。
    - 描述在 `text`（旧版/兼容 prompt）；参考实现在 `code`；官方断言在 `test_list`。
    - entry_point 从 `code` 的首个 def 解析；把 test_setup_code 的 import 拼到测试头。
    """
    task_id = f"mbpp_{row['task_id']}"
    text = row.get("text") or row.get("prompt", "") or ""
    code = row.get("code", "") or ""
    setup = (row.get("test_setup_code") or "").strip()
    asserts = row.get("test_list", []) or []

    # entry_point：从参考实现的第一个 def 行解析函数名
    entry = ""
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("def "):
            entry = s[4:].split("(")[0].strip()
            break

    # 参考测试：断言本身已是完整 `assert ... == ...` 语句，直接包装成 pytest 函数
    ref_lines = [setup] if setup else []
    for i, a in enumerate(asserts):
        a = a.strip()
        body = a if a.startswith("assert") else f"assert {a}"
        ref_lines.append(f"def test_ref_{i}():\n    {body}")
    ref = "\n\n".join(ref_lines)

    examples: list[Example] = []
    for a in asserts:
        if "==" in a:
            lhs, rhs = a.split("==", 1)
            examples.append(Example(lhs.replace("assert", "").strip(), rhs.strip()))

    return Problem(
        task_id=task_id, entry_point=entry, prompt_header=text,
        source=code, examples=examples, reference_tests=ref,
    )


def get_problems(source: str = "samples", limit: int | None = None) -> list[Problem]:
    if source == "samples":
        problems = get_sample_problems()
    elif source in ("humaneval", "mbpp"):
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "加载真实数据集需要可选依赖：pip install '.[data]'（datasets）。"
                "或先用 source='samples' 跑通流水线。"
            ) from e
        if source == "humaneval":
            ds = load_dataset("openai/openai_humaneval", split="test")
            problems = [_humaneval_to_problem(dict(r)) for r in ds]
        else:
            ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
            problems = [_mbpp_to_problem(dict(r)) for r in ds if r.get("code")]
    else:
        raise ValueError(f"未知数据源: {source}")

    return problems[:limit] if limit else problems
