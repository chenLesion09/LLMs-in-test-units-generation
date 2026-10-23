"""结果分析：读取 run_*.jsonl，产出关键对照、显著性检验与图表。

设计文档 §8 的落地：
- 配对对照：C2 vs C0（闭环净增益）、C2 vs C3（"缺失编译器"净贡献）。
- 检验：有效性用 McNemar 精确二项；变异分数/覆盖率用 Wilcoxon 符号秩；覆盖率-MS 用 Spearman。
- 图：① 有效性 & MS 聚合柱状；② token-MS 帕累托前沿；③ 覆盖率-MS 相关散点。

pandas/matplotlib/scipy 缺失时自动降级为纯文本（不报错）。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index(records: list[dict]) -> dict[tuple, dict]:
    return {(r["model"], r["condition"], r["task_id"]): r for r in records}


def _paired(idx, model, a, b):
    """返回同一题目在条件 a、b 上的配对 (task, recA, recB)。"""
    tasks = {k[2] for k in idx if k[0] == model and k[1] == a} & \
            {k[2] for k in idx if k[0] == model and k[1] == b}
    return [(t, idx[(model, a, t)], idx[(model, b, t)]) for t in sorted(tasks)]


def _mcnemar_p(b, c):
    try:
        from scipy.stats import binomtest
        if b + c == 0:
            return 1.0
        return binomtest(b, b + c, 0.5).pvalue
    except ImportError:
        return None


def _wilcoxon_p(diffs):
    try:
        from scipy.stats import wilcoxon
        nz = [d for d in diffs if d != 0]
        if len(nz) < 2:
            return None
        return wilcoxon(nz).pvalue
    except ImportError:
        return None


def text_report(records: list[dict]) -> None:
    idx = _index(records)
    for model in sorted({r["model"] for r in records}):
        print(f"\n==== 模型 {model} ====")
        for a, b, label in (("C2", "C0", "闭环 vs 零样本"), ("C2", "C3", "执行反馈 vs 自我复核")):
            pairs = _paired(idx, model, a, b)
            if not pairs:
                continue
            va = sum(1 for _, ra, rb in pairs if ra["final_status"] == "valid" and rb["final_status"] != "valid")
            vb = sum(1 for _, ra, rb in pairs if rb["final_status"] == "valid" and ra["final_status"] != "valid")
            ms_diffs, cov_diffs = [], []
            for _, ra, rb in pairs:
                sa, sb = ra["metrics"]["mutation_score"], rb["metrics"]["mutation_score"]
                if sa is not None and sb is not None:
                    ms_diffs.append(sa - sb)
            mean_ms = sum(ms_diffs) / len(ms_diffs) if ms_diffs else 0.0
            pm = _mcnemar_p(va, vb)
            pw = _wilcoxon_p(ms_diffs)
            pm_s = f", McNemar p={pm:.4f}" if pm is not None else ""
            pw_s = f", Wilcoxon p={pw:.4f}" if pw is not None else ""
            print(
                f"  {label} ({a} vs {b}): 有效性 a>b={va} b>a={vb}{pm_s} | "
                f"MS Δ={mean_ms:+.3f}{pw_s}"
            )


def plot_aggregate(records: list[dict], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = defaultdict(lambda: {"valid": [], "ms": [], "tokens": []})
    for r in records:
        k = (r["model"], r["condition"])
        agg[k]["valid"].append(1 if r["final_status"] == "valid" else 0)
        if r["metrics"]["mutation_score"] is not None:
            agg[k]["ms"].append(r["metrics"]["mutation_score"])
        agg[k]["tokens"].append(r["total_tokens"])

    models = sorted({m for m, _ in agg})
    conds = ["C0", "C1", "C2", "C3"]
    x = range(len(conds))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    offsets = [-0.2, 0, 0.2, 0.25][:len(models)] if len(models) <= 4 else [0] * len(models)
    for mi, model in enumerate(models):
        pos = [xi + offsets[mi] for xi in x]
        for ax, key, title in zip(axes, ("valid", "ms", "tokens"),
                                  ("validity rate", "mean mutation score", "mean total tokens")):
            vals = []
            for c in conds:
                d = agg.get((model, c))
                if not d or not d[key]:
                    vals.append(0)
                else:
                    vals.append(sum(d[key]) / len(d[key]))
            ax.bar(pos, vals, width=0.18, label=model)
    axes[0].set_title("Validity rate"); axes[1].set_title("Mutation score"); axes[2].set_title("Token cost")
    for ax in axes:
        ax.set_xticks(list(x)); ax.set_xticklabels(conds); ax.set_xlabel("condition")
        ax.legend(fontsize=8)
    fig.suptitle("Per-condition aggregate (validity / quality / cost)")
    fig.tight_layout()
    fig.savefig(fig_dir / "aggregate_bars.png", dpi=130)
    plt.close(fig)
    print(f"  [图] {fig_dir / 'aggregate_bars.png'}")


def plot_pareto(records: list[dict], fig_dir: Path) -> None:
    """质量-成本帕累托：按模型分面的小倍数图，半透明试次点 + 条件均值星标。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import random

    colors = {"C0": "#9e9e9e", "C1": "#4285f4", "C2": "#ea4335", "C3": "#fbbc05"}
    data = [r for r in records
            if r["metrics"]["mutation_score"] is not None and r["metrics"].get("valid_on_p")]
    models = sorted({r["model"] for r in data})
    if not models:
        return
    rnd = random.Random(42)

    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 4.4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    conds = [c for c in ("C0", "C1", "C2", "C3")]
    for ax, model in zip(axes, models):
        for c in conds:
            pts = [(r["total_tokens"], r["metrics"]["mutation_score"])
                   for r in data if r["model"] == model and r["condition"] == c]
            if not pts:
                continue
            ax.scatter([x + rnd.uniform(-120, 120) for x, _ in pts],
                       [y + rnd.uniform(-0.012, 0.012) for _, y in pts],
                       s=16, alpha=0.35, c=colors[c], edgecolors="none")
            mx = sum(x for x, _ in pts) / len(pts)
            my = sum(y for _, y in pts) / len(pts)
            ax.scatter(mx, my, s=210, marker="*", c=colors[c],
                       edgecolors="black", linewidths=0.8, zorder=5)
            ax.annotate(c, (mx, my), textcoords="offset points", xytext=(9, 7), fontsize=8)
        ax.set_title(f"{model} (n={sum(1 for r in data if r['model'] == model)})", fontsize=10)
        ax.set_xlabel("total tokens (cost)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mutation score (valid suites only)")
    axes[0].set_ylim(-0.03, 1.08)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markeredgecolor=c, label=k) for k, c in colors.items()]
    handles.append(plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="grey",
                              markeredgecolor="black", markersize=12, label="mean"))
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("Quality vs cost by model (small multiples; star = condition mean)")
    fig.tight_layout()
    fig.savefig(fig_dir / "pareto.png", dpi=130)
    plt.close(fig)
    print(f"  [图] {fig_dir / 'pareto.png'}")


def plot_cov_vs_ms(records: list[dict], fig_dir: Path) -> None:
    """覆盖率分箱箱线图 + 抖动散点：揭示同为满覆盖时 MS 仍巨幅弥散。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import random

    pairs = [(r["metrics"]["coverage_line"], r["metrics"]["mutation_score"])
             for r in records
             if r["metrics"].get("coverage_line") is not None
             and r["metrics"].get("mutation_score") is not None]
    if not pairs:
        return
    rnd = random.Random(7)

    bins = [(0, 70, "<70%"), (70, 90, "70~90%"), (90, 100, "90~<100%"), (100, 101, "=100%")]
    groups, labels = [], []
    for lo, hi, lab in bins:
        vals = [ms for cov, ms in pairs if lo <= cov < hi]
        groups.append(vals)
        labels.append(f"{lab}\n(n={len(vals)})")

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    shown = [(l, g) for l, g in zip(labels, groups) if g]
    bp = ax.boxplot([g for _, g in shown], widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.6),
                    flierprops=dict(marker=""))
    ax.set_xticks(range(1, len(shown) + 1))
    ax.set_xticklabels([l for l, _ in shown])
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8"); patch.set_alpha(0.55)
    pos = 0
    for g in groups:
        if not g:
            continue
        pos += 1
        ax.scatter([pos + rnd.uniform(-0.16, 0.16) for _ in g],
                   [v + rnd.uniform(-0.012, 0.012) for v in g],
                   s=14, alpha=0.35, c="#4285f4", edgecolors="none", zorder=4)

    rho = None
    try:
        from scipy.stats import spearmanr
        rho = spearmanr([c for c, _ in pairs], [m for _, m in pairs]).statistic
        ax.set_title(f"Mutation score by coverage bin  (Spearman rho={rho:.2f}, n={len(pairs)})")
    except ImportError:
        ax.set_title("Mutation score by coverage bin")

    # 满覆盖组的关键标注：MS 跨度
    full = groups[-1]
    if full and len(full) > 1:
        ax.annotate(f"all at 100% coverage,\nyet MS spans {min(full):.2f}~{max(full):.2f}",
                    xy=(pos, 0.5), xytext=(pos - 1.9, 0.38),
                    fontsize=9, color="#b30000",
                    arrowprops=dict(arrowstyle="->", color="#b30000", lw=1.2))
    ax.set_xlabel("line coverage bin")
    ax.set_ylabel("mutation score")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage_vs_ms.png", dpi=130)
    plt.close(fig)
    print(f"  [图] {fig_dir / 'coverage_vs_ms.png'}  (rho={rho})")


def analyze(path: Path, make_plots: bool = True) -> None:
    records = _load(path)
    print(f"分析：{path}  (n={len(records)} 试次)")
    text_report(records)
    if not make_plots:
        return
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("  （未安装 matplotlib，跳过绘图；pip install '.[stats]' 可启用）")
        return
    from src.config import DEFAULT_SETTINGS
    fig_dir = DEFAULT_SETTINGS.figures_dir
    fig_dir.mkdir(exist_ok=True)
    plot_aggregate(records, fig_dir)
    plot_pareto(records, fig_dir)
    plot_cov_vs_ms(records, fig_dir)


def latest_run(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("run_*.jsonl"))
    return files[-1] if files else None


if __name__ == "__main__":
    from src.config import DEFAULT_SETTINGS

    ap = argparse.ArgumentParser(description="分析实验结果 JSONL 并出图")
    ap.add_argument("jsonl", nargs="?", default=None, help="结果文件；缺省取 results/ 最新")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    target = Path(args.jsonl) if args.jsonl else latest_run(DEFAULT_SETTINGS.results_dir)
    if not target or not target.exists():
        raise SystemExit("未找到结果 jsonl，请先运行 main.py。")
    analyze(target, make_plots=not args.no_plots)
