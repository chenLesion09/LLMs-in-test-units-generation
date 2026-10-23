"""全局配置：模型注册表、预算、超时、窗口上限、路径。

真实密钥只从环境变量读取，绝不写入代码或仓库。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 阿里百炼（DashScope）OpenAI 兼容接入点：三模型共用同一 base_url
# ---------------------------------------------------------------------------
BAILIAN_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

# Token Plan 的密钥；环境变量注入
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")


@dataclass(frozen=True)
class ModelConfig:
    """单个被评测模型的接入配置。

    window: 该模型在百炼上的上下文窗口上限（token），用于 §7 窗口占用率。
    model_id: 百炼控制台实际可用的 model id，请按控制台核对后修改。
    """

    key: str            # 实验内部标识，如 "glm"
    display: str        # 展示名，如 "GLM-4.5"
    model_id: str       # 传给 API 的 model 字段
    window: int         # 上下文窗口上限
    max_output_tokens: int = 4096


# 注意：model_id 为占位默认值，务必以你百炼控制台的上架 id 为准替换（见设计文档 §11.1）
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "glm": ModelConfig("glm", "GLM-5.3", "glm-5.3", window=1_000_000),
    "qwen": ModelConfig("qwen", "qwen3.8-flash", "qwen3.8-flash", window=1_000_000),
    "deepseek": ModelConfig("deepseek", "deepseek-v4.1-flash", "deepseek-v4.1-flash", window=1_000_000),
}


@dataclass
class Budget:
    """单实例运行预算，用于防失控（见 §11.1）。"""

    max_rounds: int = 3            # 闭环最大轮次 R（有效性阶段）
    max_adequacy_rounds: int = 2   # 阶段二充分性修复的最大轮次（独立收紧，降本）
    per_test_timeout: float = 5.0  # 沙箱单个测试套件的墙钟超时（秒）
    max_llm_calls: int = 12        # 单实例允许的最大 LLM 往返次数（硬上限）
    max_prompt_tokens: int = 60_000  # 单轮 prompt token 软上限，超过则压缩历史
    window_hit_ratio: float = 0.9  # 占用率超过该比例即记一次"撞窗事件"


@dataclass
class Settings:
    temperature: float = 0.0
    backend: str = "mock"          # "mock" | "bailian"
    parallel: int = 1              # 按 (model, problem) 并发的工作线程数；1=串行
    budget: Budget = field(default_factory=Budget)

    # 路径
    project_root: Path = Path(__file__).resolve().parent.parent
    results_dir: Path = project_root / "results"
    figures_dir: Path = project_root / "figures"

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)


DEFAULT_SETTINGS = Settings()
