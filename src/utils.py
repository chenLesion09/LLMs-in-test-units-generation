"""通用工具函数：代码块提取、token 估算、时间戳等。"""
from __future__ import annotations

import re
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_FENCE_RE = re.compile(
    r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
# 开围栏（用于处理被截断、未闭合的代码块）
_OPEN_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n", re.IGNORECASE)


def extract_code(text: str) -> str:
    """从模型回复中提取 Python 代码。

    优先取成对的 ```python ... ``` 代码块（多个则拼接）；
    若只有开围栏（输出被 max_tokens 截断、未闭合），取第一个开围栏之后的全部内容；
    若无任何围栏，则整段返回；最后删除残留的 ``` 围栏行。
    """
    if not text:
        return ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        code = "\n\n".join(b.strip("\n") for b in blocks)
    else:
        m = _OPEN_RE.search(text)
        code = text[m.end():] if m else text
    # 保险：剔除任何遗留的围栏行（``` / ```python 等）
    lines = [
        ln for ln in code.splitlines()
        if ln.strip().lower().strip("\r") not in ("```", "```python", "```py")
    ]
    return "\n".join(lines).strip("\n") + "\n"


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（约 4 字符/token，仅用于 Mock/兜底）。

    真实调用一律采用 API 返回的 usage 作为成本真值。
    """
    if not text:
        return 0
    # 中英混排：按字符数 / 4 估计，最小为 1
    return max(1, len(text) // 4)


def truncate_traceback(stderr: str, max_frames: int = 6, max_chars: int = 1200) -> str:
    """把执行报错裁剪到"最近若干帧 + 最终错误行"，用于 §7 的窗口对策。"""
    if not stderr:
        return ""
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = lines[-max_frames:]
    text = "\n".join(tail)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text
