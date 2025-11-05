import pytest

from mapping_field.new_code.binary_expansion import BinaryExpansion
from mapping_field.new_code.mapping_field import MapElementConstant, Var, NamedFunc, InvalidInput
from mapping_field.new_code.promises import BoolVar


# from mapping_field.new_code.binary_expansion import BinaryExpansion, BoolVar
# from mapping_field.new_code.ranged_condition import RangeCondition, SingleAssignmentCondition


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var.clear_vars()
    NamedFunc.clear_vars()


def test_equality_constant():
    x1 = BinaryExpansion([1, 0, 1, 1]).simplify2()
    assert x1 == 13


def test_equality():
    x1 = BinaryExpansion([BoolVar('v1'), 0, 1]).simplify2()
    x2 = BinaryExpansion([BoolVar('v1'), 0, 1]).simplify2()
    assert x1 == x2

    x2 = BinaryExpansion([BoolVar('v1'), 0, 1, 0, 0]).simplify2()
    assert x1 == x2
    assert x2 == x1

def test_constant_split():
    x = BinaryExpansion([BoolVar('v1'), 0, 1])
    constant = 4
    pure = BinaryExpansion([BoolVar('v1')])
    assert x.split_constant() == (pure, constant)

    x = BinaryExpansion([1, 0, 1])
    constant = 5
    pure = None
    assert x.split_constant() == (pure, constant)

    x = BinaryExpansion([BoolVar('v1'), 0, BoolVar('v2')])
    constant = 0
    pure = BinaryExpansion([BoolVar('v1'), 0, BoolVar('v2')])
    assert x.split_constant() == (pure, constant)

def addition_test(x, y, x_plus_y):
    result = x + y
    assert result == x_plus_y.simplify2()
    assert result - x == y.simplify2()
    assert result - y == x.simplify2()

def test_arithmetic_constant():
    addition_test(
        BinaryExpansion([1, 0, 0, 1, 1, 1, 0, 1]),
        BinaryExpansion([0, 1, 0, 1, 1, 1, 0, 1]),
        BinaryExpansion([1, 1, 0, 0, 1, 1, 1, 0, 1])
    )

    # extra carry in the end
    addition_test(
        BinaryExpansion([1, 1, 0, 1]),
        BinaryExpansion([0, 1, 0]),
        BinaryExpansion([1, 0, 1, 1])
    )

# def test_arithmetic():
#     v1 = BoolVar('v1')
#     v2 = BoolVar('v2')
#
#     addition_test(
#         BinaryExpansion([v1, 0, 1]),
#         BinaryExpansion([0, v2, 1]),
#         BinaryExpansion([v1, v2, 0, 1])
#     )
#
#     addition_test(
#         BinaryExpansion([v1, 0, 1]),
#         BinaryExpansion([0, v2, 0, 1]),
#         BinaryExpansion([v1, v2, 1, 1])
#     )
#
#     addition_test(
#         BinaryExpansion([1, 1]),
#         BinaryExpansion([1, v1]),
#         BinaryExpansion([0, v1, 1])
#     )
#
# def test_shift():
#     v1 = BoolVar('v1')
#     v2 = BoolVar('v2')
#
#     x1 = BinaryExpansion([v1, v2, 0, 1, 1])
#     x2 = BinaryExpansion([0, 0, v1, v2, 0, 1, 1])
#     x3 = BinaryExpansion([0, 0, 0, 0, v1, v2, 0, 1, 1])
#
#     assert x1.shift(+2) == x2
#     assert x1.shift(+4) == x3
#
#     assert x2.shift(-2) == x1
#     assert x2.shift(+2) == x3
#
#     assert x3.shift(-4) == x1
#     assert x3.shift(-2) == x2
#
# def test_simplify_range_condition():
#     v = [BoolVar(f'v_{i}') for i in range(4)]
#     x = BinaryExpansion(v)  # A number in [0,16)
#
#     # Note that this should pass without the call to simplified, since it is done automatically when trying to
#     # compare the conditions. In other words, each condition here is simplified twice, though the second time should
#     # be quick since the condition already "knows" that it is simplified.
#     # The reason for that extra "simplified" call is for debugging, namely separate the "real" call to simplify
#     # and the call to compare the condition.
#     condition1 = RangeCondition(x, (3, 100)).simplify()
#     condition2 = RangeCondition(x, (3, 16)).simplify()
#     assert condition1 == condition2
#
#     x = BinaryExpansion([1] + v)  # A number in [1,32), but only odd numbers
#
#     condition1 = RangeCondition(x, (0, 10)).simplify()
#     condition2 = RangeCondition(x, (1, 10)).simplify()
#     assert condition1 == condition2
#
#     # Only odd number in [16, 19) is 17, which is the only integer in [17,18)
#     condition1 = RangeCondition(x, (16, 19)).simplify()
#     condition2 = RangeCondition(x, (17, 18)).simplify()
#     assert condition1 == condition2
#
#     condition1 = RangeCondition(x, (17, 100)).simplify()
#     condition2 = RangeCondition(v[3], (1, 2)).simplify()
#     assert condition1 == condition2
#
#     condition2 = SingleAssignmentCondition(v[3] , 1)
#     assert condition1 == condition2
#
# def test_sum_of_bool():
#     x, y = BoolVar('x'), BoolVar('y')
#     condition = (0 < (x+y)-1)
#     condition = condition.simplify()
#
#     result = (x << 1) & (y << 1)
#
#     assert condition == result
#
# def test_binary_simplifier():
#     xx = [BoolVar(f'x_{i}') for i in range(4)]
#     full_bin = BinaryExpansion(xx)
#
#     # v1 = BinaryExpansion([xx[0], xx[1]])
#     # v2 = BinaryExpansion([0, 0, xx[2], xx[3]])
#     # result = v1 + v2
#     # assert result == full_bin
#
#     v1 = BinaryExpansion([xx[0], xx[1]])
#     v2 = BinaryExpansion([xx[2], xx[3]])
#     result = v1 + 4 * v2
#     assert result == full_bin
#
#     v1 = BinaryExpansion([xx[0], xx[1], xx[2]])
#     result = v1 + 8 * xx[3]
#     assert result == full_bin
#
#     v1 = BinaryExpansion([xx[1], xx[2], xx[3]])
#     result = 2 * v1 + xx[0]
#     assert result == full_bin
#


