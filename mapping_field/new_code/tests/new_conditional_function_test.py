import logging

import pytest

from mapping_field.log_utils.tree_loggers import TreeLogger
from mapping_field.new_code.conditional_function import ConditionalFunction
from mapping_field.new_code.mapping_field import MapElementConstant, Var
from mapping_field.new_code.tests.utils import DummyCondition, DummyMap

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def reset_logs():
    TreeLogger.reset()

def test_simple_construction():
    dummy = DummyMap(0)
    cond_func = ConditionalFunction([
        (dummy < 0, dummy)
    ])
    assert str(cond_func) == '[ DummyMap(0)<0 -> DummyMap(0) ]'

def test_post_generation_independence():
    dummy0 = DummyMap(0)
    x = Var('x')
    regions = [
        (dummy0 < 0, x),
        (x < 0, dummy0),
    ]
    func = ConditionalFunction(regions)
    assert str(func) == '[ DummyMap(0)<0 -> x  ;  x<0 -> DummyMap(0) ]'

    # Changing the region list should not change the function
    dummy1 = DummyMap(1)
    regions[0] = (dummy0 < 0, dummy1)
    regions[1] = (dummy1 < 0, dummy0)
    region_changed_func = ConditionalFunction(regions)

    assert str(func) == '[ DummyMap(0)<0 -> x  ;  x<0 -> DummyMap(0) ]'
    assert str(region_changed_func) == '[ DummyMap(0)<0 -> DummyMap(1)  ;  DummyMap(1)<0 -> DummyMap(0) ]'

    # Calling the function
    assigned_func = func({x: dummy1})

    assert str(assigned_func) == '[ DummyMap(0)<0 -> DummyMap(1)  ;  DummyMap(1)<0 -> DummyMap(0) ]'
    # Some indication that func is frozen
    assert str(func) == '[ DummyMap(0)<0 -> x  ;  x<0 -> DummyMap(0) ]'

def test_evaluate():
    dummy_conditions = [DummyCondition(i) for i in range(3)]

    func = ConditionalFunction([
        (dummy_conditions[0], MapElementConstant(5)),
        (dummy_conditions[1], MapElementConstant(5)),
        (dummy_conditions[2], MapElementConstant(5)),
    ])
    assert func.evaluate() == 5

    func = ConditionalFunction([
        (dummy_conditions[0], MapElementConstant(3)),
        (dummy_conditions[1], MapElementConstant(5)),
        (dummy_conditions[2], MapElementConstant(5)),
    ])
    assert func.evaluate() is None

    func = ConditionalFunction([
        (dummy_conditions[0], MapElementConstant(3)),
        (dummy_conditions[1], DummyMap(0)),
    ])
    assert func.evaluate() is None

def test_equality_to_standard_function():
    dummy_conditions = [DummyCondition(i) for i in range(3)]
    dummy = DummyMap(0)

    cond_func = ConditionalFunction([
        (dummy_conditions[0], dummy),
        (dummy_conditions[1], dummy),
        (dummy_conditions[2], dummy),
    ])

    assert cond_func.simplify2() is dummy

    cond_func = ConditionalFunction([
        (dummy_conditions[0], dummy),
        (dummy_conditions[1], dummy),
        (dummy_conditions[2], 0),
    ])

    assert cond_func.simplify2() != dummy


def test_equality_to_conditional_function():

    dummies = [DummyCondition(values={i}) for i in range(3)]  # Two distinct dummies do not intersect
    cc = [MapElementConstant(i) for i in range(3)]

    cond_func1 = ConditionalFunction([
        (dummies[0], cc[0]),
        (dummies[1], cc[1]),
        (dummies[2], cc[2]),
    ])
    cond_func2 = ConditionalFunction([
        (dummies[0], cc[0]),
        (dummies[1], cc[1]),
        (dummies[2], cc[2]),
    ])

    assert cond_func1 == cond_func2
#
# def test_combining_regions():
#     x, y = BoolVar('x'), BoolVar('y')
#
#     func = ConditionalFunction([
#         ( (x << 0) & (y << 0) , MapElementConstant(0)),
#         ( (x << 1) & (y << 0) , MapElementConstant(1)),
#         ( (y << 1) , x),
#     ])
#
#     func = func.simplify2()
#     assert func == x
#
#
# def test_equality_region_wise():
#     x = Var('x')
#
#     cond_func = ConditionalFunction([
#         (SingleAssignmentCondition(x, 0), MapElementConstant(0)),
#         (SingleAssignmentCondition(x, 2), MapElementConstant(2)),
#         (RangeCondition(x, (7, 17)), x),
#     ])
#
#     assert cond_func == x
#
#
# def test_addition():
#     dummies = [DummyCondition(i) for i in range(5)]
#
#     cond_func1 = ConditionalFunction([
#         (dummies[0] | dummies[1], MapElementConstant(0)),
#         (dummies[2], MapElementConstant(10))
#     ])
#
#     cond_func2 = ConditionalFunction([
#         (dummies[0], MapElementConstant(100)),
#         (dummies[1] | dummies[2], MapElementConstant(200))
#     ])
#
#     cond_add = cond_func1 + cond_func2
#
#     result = ConditionalFunction([
#         (dummies[0], MapElementConstant(100)),
#         (dummies[1], MapElementConstant(200)),
#         (dummies[2], MapElementConstant(210))
#     ])
#
#     assert result == cond_add, f'could not match:\n{result}\n{cond_add}'
#
#
# def test_addition_with_ranges():
#     dummy_map = DummyMap(0)
#
#     def ranged(low, high):
#         return RangeCondition(dummy_map, (low, high))
#
#     cond_func1 = ConditionalFunction([
#         (ranged(0,10), MapElementConstant(0)),
#         (ranged(10,30), MapElementConstant(10))
#     ])
#
#     cond_func2 = ConditionalFunction([
#         (ranged(0,20), MapElementConstant(100)),
#         (ranged(20,30), MapElementConstant(200))
#     ])
#
#     cond_add = cond_func1 + cond_func2
#
#     result = ConditionalFunction([
#         (ranged(10,20), MapElementConstant(110)),
#         (ranged(0,10), MapElementConstant(100)),
#         (ranged(20,30), MapElementConstant(210))
#     ])
#
#     assert  result == cond_add
#
#
# def test_simplification():
#
#     # combine regions with the same function
#     dummy_cond = [DummyCondition(i) for i in range(5)]
#     dummy_func = [DummyMap(i) for i in range(5)]
#
#     cond_func = ConditionalFunction([
#         (dummy_cond[0], dummy_func[0]),
#         (dummy_cond[1], dummy_func[1]),
#         (dummy_cond[2], dummy_func[0]),
#     ])
#     cond_func = cond_func.simplify2()
#
#     simplified = ConditionalFunction([
#         (dummy_cond[0] | dummy_cond[2], dummy_func[0]),
#         (dummy_cond[1], dummy_func[1]),
#     ])
#
#     assert cond_func == simplified
#
#     # Combine regions with assignemtns
#     x = Var('x')
#     xx = Linear.of(x)
#
#     cond_func = ConditionalFunction([
#         (RangeCondition(x, (0, 10)), xx + 3),
#         (SingleAssignmentCondition(x, 10), MapElementConstant(13)),
#     ])
#
#     cond_func = cond_func.simplify2()
#
#     assert cond_func == xx + 3
#
#
# def test_linear_ranged_condition_subtraction():
#     vv = [BoolVar(f'x_{i}') for i in range(4)]
#     x = BinaryExpansion(vv)
#     xx = Linear.of(x)
#
#     v1 = ReLU(xx-7)
#     v2 = ReLU(xx-8)
#     v = v1 - v2
#     v = v.simplify2()
#
#     # TODO: improve union \ intersection of conditions
#
#     assert v == x.coefficients[3]
#
#     v = 8 * v
#     u = xx - v
#
#     result = BinaryExpansion(vv[:3])
#     assert u == result
#
#
# def test_general_assignment():
#     x = Linear.of(Var('x'))
#
#     func = ConditionalFunction([
#         (x<0, -x),
#         (x>=0, x + 7)
#     ])
#
#     condition1 = (func.where() == 10)
#     condition2 = (x.where() == -10) | (x.where() == 3)
#     assert condition1 == condition2
#
#
# def test_comparisons():
#     x = Linear.of(Var('x'))
#
#     func = ConditionalFunction([
#         (x<0, -x),
#         (x>=0, x + 7)
#     ])
#
#     condition1 = (func <= 10)
#     condition2 = RangeCondition(x, (-10,4))
#     assert condition1 == condition2
#
#     condition1 = (func > 10)
#     condition2 = (x<-10) | (4<=x)
#     assert condition1 == condition2
#
#     condition1 = (func >= 10)
#     condition2 = (x<-9) | (3<=x)
#     assert condition1 == condition2
#
#     condition1 = (func < 10)
#     condition2 = RangeCondition(x, (-9,3))
#     assert condition1 == condition2
#
#
# def test_mul_generation():
#     x = Linear.of(Var('x'))
#
#     func1 = (3<=x) * x
#
#     func2 = ConditionalFunction([
#         (3<=x, x),
#         (x<3, 0)
#     ])
#
#     assert func1 == func2
#
