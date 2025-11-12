from typing import Optional

from mapping_field.binary_expansion import BinaryExpansion
from mapping_field.conditions import BinaryCondition, FalseCondition, TrueCondition
from mapping_field.log_utils.tree_loggers import TreeLogger, blue
from mapping_field.mapping_field import MapElement, VarDict
from mapping_field.promises import IsCondition
from mapping_field.ranged_condition import BoolVar

simplify_logger = TreeLogger(__name__)


def test_simplify_range():
    vv = [BoolVar(f"x_{i}") for i in range(4)]  # a number in [0,15]
    x = BinaryExpansion(vv)
    x3 = BinaryExpansion(vv[:3])

    cond1 = (x <= 7).simplify2()
    cond2 = (vv[3] << 0).simplify2()
    assert cond1 == cond2

    cond1 = (x <= 6).simplify2()
    cond2 = ((vv[3] << 0) & (x3 <= 6)).simplify2()
    assert cond1 == cond2

    cond1 = (x >= 8).simplify2()
    cond2 = (vv[3] << 1).simplify2()
    assert cond1 == cond2

    cond1 = (x >= 9).simplify2()
    cond2 = ((vv[3] << 1) & (1 <= x3)).simplify2()
    assert cond1 == cond2

    # TODO: The following doesn't work. Think if I should and then how to implement it.
    # cond1 = (x <= 8).simplify2()
    # cond2 = ((vv[3] << 0) | (x << 8)).simplify2()
    # assert cond1 == cond2

    # cond1 = (x >= 7).simplify2()
    # cond2 = ((vv[3] << 1) | (x << 7)).simplify2()
    # assert cond1 == cond2


def test_extend_range_to_full():
    # Ranged conditions on binary expansion get simplified to their boolean variables.
    # Make sure that they can recombined back together
    vv = [BoolVar(f"x_{i}") for i in range(3)]
    x = BinaryExpansion(vv)

    cond1 = (x < 8).simplify2()
    cond2 = (x >= 0).simplify2()
    assert cond1 is TrueCondition
    assert cond2 is TrueCondition

    for k in range(0, 8):
        simplify_logger.tree.reset()
        simplify_logger.log(f"Running test on {blue(k)}")
        cond1 = (x < k).simplify2()
        cond2 = (k <= x).simplify2()
        assert cond1 | cond2 is TrueCondition, f"Could not combine Bin<{k} | {k}<=Bin"


#
# def test_extend_range_partially():
#     vv = [BoolVar(f'x_{i}') for i in range(4)]
#     x = BinaryExpansion(vv)
#
#     def from_points(a: int, b: int, c: int):
#         cond1 = RangeCondition(x, (a,b))
#         cond2 = RangeCondition(x, (b,c))
#         result = RangeCondition(x, (a,c))
#         assert cond1 | cond2 == result
#         assert cond2 | cond1 == result
#
#     from_points(1,7,13)
#     from_points(1,8,13)
#     from_points(1,9,13)

#
# def test_extend_range_by_assignment():
#     vv = [BoolVar(f'x_{i}') for i in range(4)]
#     x = BinaryExpansion(vv)
#
#     cond1 = (x < 6)
#     for i in range(6, 19):
#         cond2 = x.as_assignment(i)
#         next_cond = (x < i+1)
#
#         union = cond1 | cond2
#         assert union == next_cond
#         union = cond2 | cond1
#         assert union == next_cond
#         cond1 = next_cond
