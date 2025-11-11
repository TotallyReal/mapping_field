from typing import Optional

from mapping_field.log_utils.tree_loggers import TreeLogger, blue
from mapping_field.binary_expansion import BinaryExpansion
from mapping_field.conditions import BinaryCondition, FalseCondition, TrueCondition
from mapping_field.mapping_field import MapElement, VarDict
from mapping_field.promises import IsCondition
from mapping_field.ranged_condition import BoolVar

simplify_logger = TreeLogger(__name__)


def test_assignment():
    x = BoolVar('x')

    cond = (x<<0)

    assert cond is not TrueCondition
    assert cond is not FalseCondition

    cond0 = cond(0)
    cond1 = cond(1)

    assert cond0 is TrueCondition
    assert cond1 is FalseCondition

def test_union_assignment():
    x, y = BoolVar('x'), BoolVar('y')
    simplify_logger.tree.set_active(False)
    cond = (x << 0) & (y << 0)
    simplify_logger.tree.set_active(True)
    cond = cond({x:0, y:0})
    cond = cond.simplify2()
    assert cond is TrueCondition

def test_avoid_loop_in__two_bool_vars_simplifier():
    x = BoolVar('x')
    cond = TrueCondition & (x<<0)


def test_two_var_simplifier():

    x, y = BoolVar('x'), BoolVar('y')
    functions = [
        TrueCondition,
        FalseCondition
    ]
    for v in [x,y]:
        for value in (0,1):
            functions.append(v<<value)
    for value_x,value_y in ( (0,0), (0,1), (1,0), (1,2) ):
        functions.append( (x << value_x) & (y << value_y) )
        functions.append( (x << value_x) | (y << value_y) )

    class TwoVar(MapElement):
        def __init__(self, function):
            super().__init__(variables=[x,y])
            self.promises.add_promise(IsCondition)
            self.function = function

        def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
            output_value = self.function(var_dict).simplify2()
            if isinstance(output_value, BinaryCondition):
                return output_value
            return None

    for function in functions:
        simplify_logger.tree.reset()
        simplify_logger.log(f'Running test on {blue(function)}')
        two_var = TwoVar(function)
        assert function == two_var.simplify2(), f'Failed with the function {function}'

def test_two_var_simplifier2():

    x, y = BoolVar('x'), BoolVar('y')

    cond = (x << 0) & ( (x<<0) | (y<<0) )
    cond = cond.simplify2()
    assert cond  == (x<<0)


def test_simplify_range():
    vv = [BoolVar(f'x_{i}') for i in range(4)] # a number in [0,15]
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
    cond2 = ((vv[3] << 1) & (1 <= x3 )).simplify2()
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
    vv = [BoolVar(f'x_{i}') for i in range(3)]
    x = BinaryExpansion(vv)

    cond1 = (x <  8).simplify2()
    cond2 = (x >= 0).simplify2()
    assert cond1 is TrueCondition
    assert cond2 is TrueCondition

    for k in range(0,8):
        simplify_logger.tree.reset()
        simplify_logger.log(f'Running test on {blue(k)}')
        cond1 = (x < k).simplify2()
        cond2 = (k <= x).simplify2()
        assert cond1 | cond2 is TrueCondition, f'Could not combine Bin<{k} | {k}<=Bin'

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