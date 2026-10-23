"""AST 级变异算子（确定性、可复现）。

每"一个变异体 = 对正确实现做一处最小改动"。遍历顺序固定（ast.walk），
生成的变异体列表对同一份源码稳定，满足论文可复现要求。
"""
from __future__ import annotations

import ast
import copy
from typing import Callable

from src.models import Mutant

# 关系运算的替换目标
_CMP: dict[type, list[type]] = {
    ast.Eq: [ast.NotEq],
    ast.NotEq: [ast.Eq],
    ast.Lt: [ast.LtE, ast.Gt],
    ast.LtE: [ast.Lt, ast.GtE],
    ast.Gt: [ast.GtE, ast.Lt],
    ast.GtE: [ast.Gt, ast.LtE],
    ast.In: [ast.NotIn],
    ast.NotIn: [ast.In],
    ast.Is: [ast.IsNot],
    ast.IsNot: [ast.Is],
}
# 算术运算的替换目标
_BIN: dict[type, list[type]] = {
    ast.Add: [ast.Sub, ast.Mult],
    ast.Sub: [ast.Add],
    ast.Mult: [ast.Add, ast.FloorDiv],
    ast.Div: [ast.Mult, ast.FloorDiv],
    ast.Mod: [ast.FloorDiv],
    ast.FloorDiv: [ast.Mult],
}


def _name(op: ast.AST) -> str:
    return type(op).__name__


class _Plan:
    """单个变异候选：如何改、如何描述。"""

    def __init__(self, apply_fn: Callable[[ast.AST], None], desc: str) -> None:
        self.apply = apply_fn
        self.desc = desc


def _candidates(tree: ast.AST) -> list[tuple[int, _Plan]]:
    """按稳定遍历顺序，返回 (node_index, plan) 列表。"""
    nodes = list(ast.walk(tree))
    plans: list[tuple[int, _Plan]] = []
    for idx, node in enumerate(nodes):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = node.ops[0]
            for new in _CMP.get(type(op), []):
                _new, _desc = new, _name(op)
                plans.append((idx, _Plan(
                    lambda n, _new=_new: setattr(n, "ops", [_new()]),
                    f"Compare:{_desc}->{_name(_new)}",
                )))
        elif isinstance(node, ast.BinOp):
            op = node.op
            for new in _BIN.get(type(op), []):
                _new, _desc = new, _name(op)
                plans.append((idx, _Plan(
                    lambda n, _new=_new: setattr(n, "op", _new()),
                    f"BinOp:{_desc}->{_name(_new)}",
                )))
        elif isinstance(node, ast.BoolOp):
            new = ast.Or if isinstance(node.op, ast.And) else ast.And
            _desc = _name(node.op)
            plans.append((idx, _Plan(
                lambda n, _new=new: setattr(n, "op", _new()),
                f"BoolOp:{_desc}->{_name(new)}",
            )))
        elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            for delta in (1, -1):
                _v = node.value + delta
                plans.append((idx, _Plan(
                    lambda n, _v=_v: setattr(n, "value", _v),
                    f"Const:{node.value}->{_v}",
                )))
        elif isinstance(node, ast.Return) and node.value is not None:
            plans.append((idx, _Plan(
                lambda n: setattr(n, "value", ast.Constant(value=None)),
                "Return->None",
            )))
    return plans


def enumerate_mutants(source: str) -> list[Mutant]:
    """生成该源码的所有变异体（每处一个改动）。"""
    base = ast.parse(source)
    plans = _candidates(base)
    mutants: list[Mutant] = []
    for k, (idx, plan) in enumerate(plans):
        tree = copy.deepcopy(base)
        nodes = list(ast.walk(tree))
        target = nodes[idx]
        plan.apply(target)
        try:
            ast.fix_missing_locations(tree)
            new_src = ast.unparse(tree)
            ast.parse(new_src)  # 自查
        except (SyntaxError, ValueError):
            continue
        mutants.append(Mutant(index=k, source=new_src, description=plan.desc))
    return mutants
