"""实验入口 CLI。

示例：
  # 离线 Mock，打通全流程（不消耗 API 额度）
  python main.py --backend mock --conditions C0,C1,C2,C3 --models glm,qwen,deepseek --limit 6

  # 真实百炼调用前，先设好环境变量 DASHSCOPE_API_KEY
  python main.py --backend bailian --conditions C0,C2 --models glm --source humaneval --limit 30 --rounds 3
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from src.config import Settings, Budget, MODEL_REGISTRY
from src.models import Condition
from src.runner import run_experiment


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM 闭环测试用例生成实验")
    ap.add_argument("--backend", choices=["mock", "bailian"], default="mock")
    ap.add_argument("--models", default="glm,qwen,deepseek", help="逗号分隔的 model key")
    ap.add_argument("--conditions", default="C0,C1,C2,C3", help="逗号分隔的条件")
    ap.add_argument("--source", default="samples",
                    help="samples | humaneval | mbpp | all(=humaneval+mbpp)，也可逗号分隔多个")
    ap.add_argument("--limit", type=int, default=None, help="每个数据集的题目数上限；不传或 0 = 全量")
    ap.add_argument("--rounds", type=int, default=3, help="闭环最大轮次 R")
    ap.add_argument("--parallel", type=int, default=1, help="按 (model,problem) 并发线程数；1=串行")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：复用 results/run_*.jsonl 已完成任务，只补跑缺失部分")
    ap.add_argument("--temperature", type=float, default=0.0)
    return ap.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    settings = Settings(
        temperature=args.temperature,
        backend=args.backend,
        parallel=max(1, args.parallel),
        budget=Budget(max_rounds=args.rounds),
    )

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(f"未知 model key: {unknown}；可选：{list(MODEL_REGISTRY)}")

    conditions = [Condition(c.strip().upper()) for c in args.conditions.split(",") if c.strip()]

    # 解析数据源：all -> humaneval+mbpp；支持逗号分隔
    raw_sources = [s.strip() for s in args.source.split(",") if s.strip()]
    sources: list[str] = []
    for s in raw_sources:
        if s == "all":
            sources += ["humaneval", "mbpp"]
        elif s in ("samples", "humaneval", "mbpp"):
            sources.append(s)
        else:
            raise SystemExit(f"未知 --source: {s}；可选 samples/humaneval/mbpp/all")
    sources = list(dict.fromkeys(sources))  # 去重保序
    limit = args.limit if args.limit else None  # 0/None 均视为全量

    print(
        f"backend={settings.backend} models={model_keys} conditions={[c.value for c in conditions]} "
        f"source={sources} limit={limit} R={args.rounds} T={args.temperature}\n"
    )
    run_experiment(settings, conditions, model_keys, source=sources, limit=limit, resume=args.resume)


if __name__ == "__main__":
    main()
