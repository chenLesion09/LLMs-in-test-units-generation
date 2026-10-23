"""沙箱执行器（提速版）。

核心变化：
- 变异体评估不再"每个变异体冷启动一次 pytest"，而是**单个进程 exec 批量跑完全部变体**
  （一次 python 启动即可，通常 <0.3s），把 O(变异体数) 次进程降到 O(1) 次。
- 不再依赖 pytest 做通过/失败判定；用纯 python 逐个调用 test_* 函数，区分
  pass / AssertionError(fail) / 其它异常(error)，并收集 traceback 供修复回喂。
- 若整批超时（如生成代码里有死循环），**回退为逐变体独立进程**执行，保证不互相牵连。
- 覆盖率仍用 pytest-cov（只在最终指标那一次调用，非热路径）。

安全说明：会 exec 模型生成的代码；用独立临时目录 + 子进程 + 墙钟超时做基本隔离，
适合可控的函数级基准；不可信来源应再叠加容器/资源限制（未来工作）。
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.models import ExecStatus

_HEADER = "from typing import List, Tuple, Dict, Optional, Any, Iterable, Set\n"

# 在子进程里运行的批量执行脚本：读取 spec.json -> 对每个 variant exec 实现后跑全部测试
_RUNNER = '''
import json, traceback, sys
spec = json.load(open("spec.json", encoding="utf-8"))
g = {"__name__": "__main__"}
try:
    exec(spec["header"], g)
    exec(spec["test_source"], g)
except SyntaxError:
    json.dump({"__error__": "syntax"}, open("result.json", "w"))
    sys.exit(0)
except Exception:
    json.dump({"__error__": "exec", "error_text": traceback.format_exc()[-2000:]},
              open("result.json", "w"))
    sys.exit(0)

tests = sorted([(n, f) for n, f in g.items() if n.startswith("test_") and callable(f)])
result = {}
for vid, src in spec["variants"]:
    p = fa = er = 0
    tbs, tout = [], []
    try:
        exec(src, g)
    except Exception:
        result[vid] = {"passed": 0, "failed": 0, "errored": 1,
                       "tests": [], "error_text": traceback.format_exc()[-2000:]}
        continue
    for name, fn in tests:
        try:
            fn(); p += 1; tout.append([name, "pass"])
        except AssertionError:
            fa += 1; tbs.append(traceback.format_exc()); tout.append([name, "fail"])
        except Exception:
            er += 1; tbs.append(traceback.format_exc()); tout.append([name, "error"])
    result[vid] = {"passed": p, "failed": fa, "errored": er,
                   "tests": tout, "error_text": "\\n".join(tbs)[-2000:]}
json.dump(result, open("result.json", "w"))
'''


@dataclass
class SuiteExec:
    status: ExecStatus
    n_tests: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_errored: int = 0
    killed: bool = False
    duration: float = 0.0
    timed_out: bool = False
    error_text: str = ""
    tests: list = field(default_factory=list)   # [(name, 'pass'|'fail'|'error'), ...]
    coverage_line: float | None = None
    coverage_branch: float | None = None


class Sandbox:
    def __init__(self, per_test_timeout: float = 5.0) -> None:
        self.timeout = per_test_timeout

    def syntax_ok(self, source: str) -> bool:
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    # ---- 批量执行（核心提速）----
    def exec_variants(
        self, test_source: str, variants: list[tuple[str, str]], *, timeout: float | None = None
    ) -> dict[str, SuiteExec]:
        if not self.syntax_ok(test_source):
            return {vid: SuiteExec(ExecStatus.SYNTAX_ERROR, error_text="SyntaxError")
                    for vid, _ in variants}

        batch_timeout = timeout if timeout is not None else max(20.0, 2.0 * len(variants))
        try:
            raw = self._run_once(test_source, variants, batch_timeout)
        except _BatchTimeout:
            # 死循环等：回退为逐变体独立进程，互不牵连
            out: dict[str, SuiteExec] = {}
            for vid, src in variants:
                out[vid] = self._run_single_fallback(test_source, src)
            return out

        if "__error__" in raw:
            st = ExecStatus.SYNTAX_ERROR if raw["__error__"] == "syntax" else ExecStatus.EXEC_ERROR
            etxt = raw.get("error_text", "")
            return {vid: SuiteExec(st, error_text=etxt) for vid, _ in variants}

        return {vid: self._to_exec(raw.get(vid, {})) for vid, _ in variants}

    def exec_single(self, p_source: str, test_source: str) -> SuiteExec:
        return self.exec_variants(test_source, [("P", p_source)],
                                   timeout=max(self.timeout, 10.0))["P"]

    def _run_once(self, test_source: str, variants: list[tuple[str, str]], timeout: float) -> dict:
        workdir = Path(tempfile.mkdtemp(prefix="sbx_"))
        try:
            spec = {
                "header": _HEADER,
                "test_source": test_source,
                "variants": [[vid, src] for vid, src in variants],
            }
            (workdir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            (workdir / "_runner.py").write_text(_RUNNER, encoding="utf-8")
            t0 = time.perf_counter()
            subprocess.run([sys.executable, "_runner.py"], cwd=workdir,
                           capture_output=True, text=True, timeout=timeout, check=False)
            dur = time.perf_counter() - t0
            res = json.loads((workdir / "result.json").read_text(encoding="utf-8"))
            for d in res.values():
                if isinstance(d, dict):
                    d["_dur"] = dur
            return res
        except subprocess.TimeoutExpired:
            raise _BatchTimeout()
        except (OSError, json.JSONDecodeError):
            raise _BatchTimeout()
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def _run_single_fallback(self, test_source: str, p_source: str) -> SuiteExec:
        workdir = Path(tempfile.mkdtemp(prefix="sbx1_"))
        try:
            spec = {"header": _HEADER, "test_source": test_source, "variants": [["P", p_source]]}
            (workdir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            (workdir / "_runner.py").write_text(_RUNNER, encoding="utf-8")
            t0 = time.perf_counter()
            try:
                subprocess.run([sys.executable, "_runner.py"], cwd=workdir,
                               capture_output=True, text=True, timeout=self.timeout, check=False)
                res = json.loads((workdir / "result.json").read_text(encoding="utf-8"))
                return self._to_exec(res.get("P", {}), time.perf_counter() - t0)
            except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
                return SuiteExec(ExecStatus.EXEC_ERROR, timed_out=True,
                                 error_text="TIMEOUT/崩溃，独立进程仍无法完成")
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _to_exec(d: dict, dur: float | None = None) -> SuiteExec:
        if not d:
            return SuiteExec(ExecStatus.EMPTY, duration=dur or 0.0)
        passed, failed, errored = d.get("passed", 0), d.get("failed", 0), d.get("errored", 0)
        n = passed + failed + errored
        killed = (failed + errored) > 0
        if errored > 0:
            st = ExecStatus.EXEC_ERROR
        elif n == 0:
            st = ExecStatus.EMPTY
        elif failed > 0:
            st = ExecStatus.FAIL_ON_P
        else:
            st = ExecStatus.VALID
        return SuiteExec(
            status=st, n_tests=n, n_passed=passed, n_failed=failed, n_errored=errored,
            killed=killed, duration=dur if dur is not None else d.get("_dur", 0.0),
            error_text=d.get("error_text", ""), tests=d.get("tests", []),
        )

    # ---- 覆盖率（仅最终指标一次，走 pytest-cov）----
    def run_with_coverage(self, p_source: str, test_source: str) -> SuiteExec:
        import re
        import shutil
        if not self.syntax_ok(test_source):
            return SuiteExec(ExecStatus.SYNTAX_ERROR, error_text="SyntaxError")
        workdir = Path(tempfile.mkdtemp(prefix="cov_"))
        try:
            (workdir / "module_under_test.py").write_text(_HEADER + p_source, encoding="utf-8")
            (workdir / "test_suite.py").write_text(
                _HEADER + "from module_under_test import *\n\n" + test_source, encoding="utf-8")
            cmd = [sys.executable, "-m", "pytest", "test_suite.py", "-q", "--no-header",
                   "-p", "no:cacheprovider", "--cov=module_under_test", "--cov-branch",
                   "--cov-report=json:coverage.json"]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                                      timeout=max(self.timeout, 15.0), check=False)
                out, err, rc = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired:
                return SuiteExec(ExecStatus.EXEC_ERROR, timed_out=True, error_text="TIMEOUT")
            dur = time.perf_counter() - t0
            cov_line = cov_branch = None
            try:
                data = json.loads((workdir / "coverage.json").read_text(encoding="utf-8"))
                tot = data.get("totals", {})
                cov_line = tot.get("percent_covered")
                nb = tot.get("num_branches") or 0
                mb = tot.get("missing_branches") or 0
                cov_branch = (100.0 * (nb - mb) / nb) if nb else None
            except (OSError, json.JSONDecodeError):
                pass
            text = out + "\n" + err

            def gv(pat):
                m = re.search(pat, text)
                return int(m.group(1)) if m else 0
            passed, failed, errored = gv(r"(\d+)\s+passed"), gv(r"(\d+)\s+failed"), gv(r"(\d+)\s+error")
            n = passed + failed + errored
            if rc in (2, 3, 4) or errored > 0:
                st = ExecStatus.EXEC_ERROR
            elif n == 0:
                st = ExecStatus.EMPTY
            elif failed > 0:
                st = ExecStatus.FAIL_ON_P
            else:
                st = ExecStatus.VALID
            return SuiteExec(status=st, n_tests=n, n_passed=passed, n_failed=failed,
                             n_errored=errored, killed=(failed + errored) > 0, duration=dur,
                             coverage_line=float(cov_line) if cov_line is not None else None,
                             coverage_branch=float(cov_branch) if cov_branch is not None else None)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class _BatchTimeout(Exception):
    pass
