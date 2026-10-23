"""离线迷你基准（6 题，函数级 Python）。

作用：在没有网络 / 未安装 datasets 时，也能完整跑通"生成-执行-变异-闭环-指标"整条流水线。
字段对齐 HumanEval：prompt_header(签名+docstring) / source(正确实现 P) / reference_tests(官方测试) / examples。
"""
from __future__ import annotations

from src.models import Example, Problem

_PROBLEMS: list[Problem] = [
    Problem(
        task_id="samp_is_even",
        entry_point="is_even",
        prompt_header=(
            "def is_even(n: int) -> bool:\n"
            '    """Return True if n is even, otherwise False."""'
        ),
        source="def is_even(n: int) -> bool:\n    return n % 2 == 0\n",
        examples=[Example("4", "True"), Example("7", "False"), Example("0", "True")],
        reference_tests=(
            "def test_even():\n    assert is_even(4) == True\n\n"
            "def test_odd():\n    assert is_even(7) == False\n\n"
            "def test_zero():\n    assert is_even(0) == True\n"
        ),
    ),
    Problem(
        task_id="samp_max_of_three",
        entry_point="max_of_three",
        prompt_header=(
            "def max_of_three(a: int, b: int, c: int) -> int:\n"
            '    """Return the largest of three integers."""'
        ),
        source=(
            "def max_of_three(a: int, b: int, c: int) -> int:\n"
            "    m = a\n"
            "    if b > m:\n        m = b\n"
            "    if c > m:\n        m = c\n"
            "    return m\n"
        ),
        examples=[Example("1, 5, 3", "5"), Example("9, 2, 4", "9"), Example("1, 1, 7", "7")],
        reference_tests=(
            "def test_b():\n    assert max_of_three(1, 5, 3) == 5\n\n"
            "def test_a():\n    assert max_of_three(9, 2, 4) == 9\n\n"
            "def test_c():\n    assert max_of_three(1, 1, 7) == 7\n"
        ),
    ),
    Problem(
        task_id="samp_count_vowels",
        entry_point="count_vowels",
        prompt_header=(
            "def count_vowels(s: str) -> int:\n"
            '    """Count vowels (a,e,i,o,u, case-insensitive) in s."""'
        ),
        source=(
            "def count_vowels(s: str) -> int:\n"
            "    count = 0\n"
            "    for ch in s.lower():\n"
            "        if ch in 'aeiou':\n"
            "            count = count + 1\n"
            "    return count\n"
        ),
        examples=[Example("'hello'", "2"), Example("'AEIOU'", "5"), Example("'xyz'", "0")],
        reference_tests=(
            "def test_hello():\n    assert count_vowels('hello') == 2\n\n"
            "def test_all():\n    assert count_vowels('AEIOU') == 5\n\n"
            "def test_none():\n    assert count_vowels('xyz') == 0\n"
        ),
    ),
    Problem(
        task_id="samp_factorial",
        entry_point="factorial",
        prompt_header=(
            "def factorial(n: int) -> int:\n"
            '    """Return n! with 0! == 1."""'
        ),
        source=(
            "def factorial(n: int) -> int:\n"
            "    result = 1\n"
            "    i = 2\n"
            "    while i <= n:\n"
            "        result = result * i\n"
            "        i = i + 1\n"
            "    return result\n"
        ),
        examples=[Example("0", "1"), Example("1", "1"), Example("5", "120")],
        reference_tests=(
            "def test_0():\n    assert factorial(0) == 1\n\n"
            "def test_1():\n    assert factorial(1) == 1\n\n"
            "def test_5():\n    assert factorial(5) == 120\n"
        ),
    ),
    Problem(
        task_id="samp_sum_list",
        entry_point="sum_list",
        prompt_header=(
            "def sum_list(nums: list) -> int:\n"
            '    """Return the sum of a list of integers."""'
        ),
        source=(
            "def sum_list(nums: list) -> int:\n"
            "    total = 0\n"
            "    for x in nums:\n"
            "        total = total + x\n"
            "    return total\n"
        ),
        examples=[Example("[1, 2, 3]", "6"), Example("[]", "0"), Example("[10]", "10")],
        reference_tests=(
            "def test_sum():\n    assert sum_list([1, 2, 3]) == 6\n\n"
            "def test_empty():\n    assert sum_list([]) == 0\n\n"
            "def test_single():\n    assert sum_list([10]) == 10\n"
        ),
    ),
    Problem(
        task_id="samp_abs_value",
        entry_point="abs_value",
        prompt_header=(
            "def abs_value(x: int) -> int:\n"
            '    """Return the absolute value of x."""'
        ),
        source=(
            "def abs_value(x: int) -> int:\n"
            "    if x < 0:\n"
            "        return -x\n"
            "    return x\n"
        ),
        examples=[Example("-5", "5"), Example("3", "3"), Example("0", "0")],
        reference_tests=(
            "def test_neg():\n    assert abs_value(-5) == 5\n\n"
            "def test_pos():\n    assert abs_value(3) == 3\n\n"
            "def test_zero():\n    assert abs_value(0) == 0\n"
        ),
    ),
]


def get_sample_problems() -> list[Problem]:
    return list(_PROBLEMS)
