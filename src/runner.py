"""实验编排：条件 × 模型 × 题目，落 JSONL 并打印聚合摘要。

支持按 (model, problem) 任务级并发（settings.parallel）：API 调用是网络 IO、
沙箱每次用独立临时目录，线程安全；变异体集在并发前一次性预构建并共享。
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.clients import get_client
from src.config import MODEL_REGISTRY, Settings
from src.dataset.loader import get_problems
from src.loop import run_trial
from src.metrics import evaluate_suite
from src.models import Condition, Problem
from src.mutation.mutate import build_non_equivalent_mutants
from src.sandbox import Sandbox
from src.utils import now_iso


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _load_sources(source: str | list[str], limit: int | None):
    """支持单个或多个数据源（如 humaneval+mbpp）合入同一次实验。limit 按每个源分别裁。"""
    sources = [source] if isinstance(source, str) else list(source)
    problems: list[Problem] = []
    dataset_of: dict[str, str] = {}
    for s in sources:
        for p in get_problems(source=s, limit=limit):
            problems.append(p)
            dataset_of[p.task_id] = s
    return problems, dataset_of


def run_experiment(
    settings: Settings,
    conditions: list[Condition],
    model_keys: list[str],
    source: str | list[str] = "samples",
    limit: int | None = None,
    resume: bool = False,
) -> list[dict]:
    settings.ensure_dirs()
    problems, dataset_of = _load_sources(source, limit)
    sandbox = Sandbox(per_test_timeout=settings.budget.per_test_timeout)

    out_path = settings.results_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    records: list[dict] = []

    # 变异体集只构建一次（与模型/条件无关），跨条件/线程共享
    mutant_cache: dict[str, list] = {}
    for p in problems:
        mutant_cache[p.task_id] = build_non_equivalent_mutants(p, sandbox)

    clients = {mk: get_client(settings, mk) for mk in model_keys}
    lock = threading.Lock()
    f = out_path.open("w", encoding="utf-8")

    tasks = [(mk, p) for mk in model_keys for p in problems]

    # 断点续跑：扫描历史 run_*.jsonl，把已凑齐全部条件的 (model,task) 任务
    # 直接复用（写入本次输出文件，使其自成为完整数据集），只补跑缺失任务。
    if resume:
        done: dict[tuple, dict] = {}
        cond_names = [c.value for c in conditions]
        valid_tasks = {(mk, p.task_id) for mk, p in tasks}
        for prev in sorted(settings.results_dir.glob("run_*.jsonl")):
            if prev == out_path:
                continue
            for line in prev.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("model"), r.get("task_id"), r.get("condition"))
                if key[0] in model_keys and (key[0], key[1]) in valid_tasks and key not in done:
                    done[key] = r

        def _is_complete(mk: str, tid: str) -> bool:
            return all((mk, tid, c) in done for c in cond_names)

        seeded = 0
        kept_tasks = []
        for mk, p in tasks:
            if _is_complete(mk, p.task_id):
                for c in cond_names:
                    rec = done[(mk, p.task_id, c)]
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    records.append(rec)
                    seeded += 1
            else:
                kept_tasks.append((mk, p))
        f.flush()
        tasks = kept_tasks
        print(f"[resume] 复用历史试次 {seeded} 个；剩余待跑任务 {len(tasks)} 个\n")

    def _emit(cond_records: list[dict]) -> None:
        with lock:
            for log in cond_records:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
                f.flush()
                records.append(log)
                _print_trial(log)

    def do_task(mk: str, p: Problem) -> list[dict]:
        cfg = MODEL_REGISTRY[mk]
        client = clients[mk]
        out: list[dict] = []
        for cond in conditions:
            try:
                trial = run_trial(
                    problem=p, condition=cond, client=client, sandbox=sandbox,
                    budget=settings.budget, window=cfg.window,
                    temperature=settings.temperature, mutants=mutant_cache[p.task_id],
                )
                trial.metrics = evaluate_suite(p, trial.final_suite, mutant_cache[p.task_id], sandbox)
                log = trial.to_log()
                log["backend"] = settings.backend
                log["dataset"] = dataset_of[p.task_id]
                log["model_id"] = cfg.model_id
                log["ts"] = now_iso()
                out.append(log)
            except Exception as e:  # 单条件 API/运行异常（429/网络等）：记录后继续，不炸掉整场实验
                out.append(_error_log(mk, p, cond, cfg, dataset_of[p.task_id], settings.backend, e))
                print(f"[{mk}|{cond.value}|{p.task_id}] !! 异常已隔离：{str(e)[:140]}")
        return out

    workers = max(1, min(settings.parallel, len(tasks))) if tasks else 1
    try:
        if workers == 1:
            for mk, p in tasks:
                _emit(do_task(mk, p))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(do_task, mk, p) for mk, p in tasks]
                for fut in as_completed(futs):
                    _emit(fut.result())
    finally:
        f.close()

    print(f"\n完整日志已写入：{out_path}\n")
    _print_aggregate(records)
    return records


def _error_log(mk, p, cond, cfg, dataset, backend, err) -> dict:
    """API/运行异常时的占位记录，保持与正常日志相同的 schema。"""
    return {
        "model": mk, "condition": cond.value, "task_id": p.task_id, "rounds": [],
        "llm_calls": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0,
        "total_tokens": 0, "peak_prompt_tokens": 0, "window_hit_events": 0,
        "final_status": "api_error", "final_suite": "",
        "metrics": {"syntax_ok": False, "executable": False, "valid_on_p": False,
                    "n_tests": 0, "n_passed": 0, "n_failed": 0,
                    "total_mutants": 0, "killed_mutants": 0,
                    "mutation_score": None, "coverage_line": None, "coverage_branch": None},
        "backend": backend, "dataset": dataset, "model_id": cfg.model_id,
        "ts": now_iso(), "error": str(err)[:200],
    }


def _print_trial(log: dict) -> None:
    m = log["metrics"]
    ms = m.get("mutation_score")
    print(
        f"[{log['model']}|{log['condition']}|{log['task_id']}] "
        f"status={log['final_status']} valid={m.get('valid_on_p')} "
        f"MS={('%.2f' % ms) if ms is not None else 'NA'} "
        f"cov={m.get('coverage_line')} rounds={len(log['rounds'])} "
        f"calls={log['llm_calls']} tokens={log['total_tokens']} peak={log['peak_prompt_tokens']}"
    )


def _print_aggregate(records: list[dict]) -> None:
    from collections import defaultdict

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["model"], r["condition"])].append(r)

    header = f"{'model':<12}{'cond':<6}{'valid率':<10}{'均MS':<10}{'cov_line':<10}{'calls':<8}{'tokens':<10}{'peak':<8}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for (mk, cond), rs in sorted(buckets.items()):
        valid = sum(1 for r in rs if r["metrics"].get("valid_on_p")) / len(rs)
        ms = _mean([r["metrics"].get("mutation_score") for r in rs])
        cov = _mean([r["metrics"].get("coverage_line") for r in rs])
        calls = _mean([r["llm_calls"] for r in rs])
        tok = _mean([r["total_tokens"] for r in rs])
        peak = _mean([r["peak_prompt_tokens"] for r in rs])
        fmt = lambda v, p=2: ("%%.%df" % p) % v if v is not None else "NA"
        print(
            f"{mk:<12}{cond:<6}{fmt(valid):<10}{fmt(ms):<10}{fmt(cov,1):<10}"
            f"{fmt(calls,1):<8}{fmt(tok,0):<10}{fmt(peak,0):<8}"
        )
    print("=" * len(header))
