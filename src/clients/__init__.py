"""客户端工厂：按 backend + model_key 产出对应 LLMClient。"""
from __future__ import annotations

import os

from src.clients.base import LLMClient
from src.clients.bailian import BailianClient
from src.clients.mock import MockClient
from src.config import BAILIAN_BASE_URL, MODEL_REGISTRY, Settings


def get_client(settings: Settings, model_key: str) -> LLMClient:
    cfg = MODEL_REGISTRY[model_key]
    if settings.backend == "mock":
        return MockClient(model_key=cfg.key, display=cfg.display)
    if settings.backend == "bailian":
        # 调用时实时读环境变量：确保 main.py 里 load_dotenv() 之后 .env 生效
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        base_url = os.environ.get("DASHSCOPE_BASE_URL", BAILIAN_BASE_URL)
        return BailianClient(cfg, base_url, api_key)
    raise ValueError(f"未知 backend: {settings.backend}")


__all__ = ["LLMClient", "get_client"]
