import pytest
from typing import List

from mapping_field.binary_expansion import BoolVar, BinaryExpansion
from mapping_field.linear import Linear
from mapping_field.conditions import FalseCondition, TrueCondition, Condition, ConditionalFunction
from mapping_field.ranged_condition import RangeCondition, SingleAssignmentCondition, ReLU
from mapping_field.mapping_field import MapElement, Var

@pytest.fixture(autouse=True)
def reset_static_variables():
    Var.clear_vars()

class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([])
        self.value = value

    def to_string(self, vars_str_list: List[str]):
        return f'DummyMap({self.value})'

    def __eq__(self, other):
        return isinstance(other, DummyMap) and other.value == self.value

def test_linear_generation():

    dummy = DummyMap(0)
    linear_dummy = Linear.of(dummy)

    func = 5*linear_dummy
    result = Linear(5, dummy, 0)
    assert func == result

    func = linear_dummy + 7
    result = Linear(1, dummy, 7)
    assert func == result

    func = 5*linear_dummy + 7
    result = Linear(5, dummy, 7)
    assert func == result

    func = 0*linear_dummy + 7
    func = func.simplify2()
    result = 7
    assert func == result

def test_linear_arithmetic():
    dummy = Linear.of(DummyMap(0))

    func1 = 5*dummy + 3
    func2 = 11*dummy + 7

    func = func1 + func2
    result = 16*dummy + 10
    assert func == result

    func = func1 - func2
    result = -6*dummy - 4
    assert func == result

# ============================== ranged condition ==============================

def test_linear_ranged_condition():
    dummy = DummyMap(0)
    linear_dummy = Linear.of(dummy)

    def ranged_condition(a, b, low, high) -> Condition:
        func = a*linear_dummy + b
        condition = RangeCondition(func, (low, high))
        condition = condition.simplify()
        return condition

    condition = ranged_condition(a=2, b=3, low=5, high=15)
    result = RangeCondition(dummy, (1,6))
    assert condition == result

    # test negative coefficient
    condition = ranged_condition(a=-2, b=3, low=5, high=15)
    result = RangeCondition(dummy, (-6,-1))
    assert condition == result

    # test zero coefficient - False
    condition = ranged_condition(a=0, b=3, low=5, high=15)
    result = FalseCondition
    assert condition == result

    # test zero coefficient - True
    condition = ranged_condition(a=0, b=7, low=5, high=15)
    result = TrueCondition
    assert condition == result


def test_assignment_condition():
    Var.clear_vars()
    x, y, z = Var('x'), Var('y'), Var('z')

    # disjoint
    cond1 = SingleAssignmentCondition.from_dict({x:1, y:2})
    cond2 = SingleAssignmentCondition.from_dict({z:3})
    prod = cond1 * cond2
    result = SingleAssignmentCondition.from_dict({x:1, y:2, z:3})
    assert prod == result

    # valid intersection
    cond1 = SingleAssignmentCondition.from_dict({x:1, y:2})
    cond2 = SingleAssignmentCondition.from_dict({y:2, z:3})
    prod = cond1 * cond2
    result = SingleAssignmentCondition.from_dict({x:1, y:2, z:3})
    assert prod == result

    # invalid intersection
    cond1 = SingleAssignmentCondition.from_dict({x:1, y:2})
    cond2 = SingleAssignmentCondition.from_dict({y:5, z:3})
    prod = cond1 * cond2
    result = FalseCondition
    assert prod == result


# def test_assignment_from_range():
#     x = BinaryExpansion.generate('x', 4)
#
#     condition = RangeCondition(x, (5,6))
#     condition = condition.simplify()
#     result = AssignmentCondition({x: 5})
#     assert condition == result
#
#     linear_func = Linear(5, x, 2)
#     condition = RangeCondition(linear_func, (7,12))
#     condition = condition.simplify()
#     result = AssignmentCondition({x: 1})
#     assert condition == result

def test_linear_shift():
    v1 = BoolVar('v1')
    v2 = BoolVar('v2')

    x1 = Linear.of(BinaryExpansion([v1, v2, 0, 1, 1]))
    x2 = Linear.of(BinaryExpansion([0, 0, v1, v2, 0, 1, 1]))
    x3 = Linear.of(BinaryExpansion([0, 0, 0, 0, v1, v2, 0, 1, 1]))

    assert x1*4 == x2
    assert x1*16 == x3
    assert x2*4 == x3


def addition_test(x, y, x_plus_y):
    addition = x + y
    assert addition == x_plus_y
    difference = x_plus_y - x
    assert difference.simplify2() == y.simplify2()
    difference = x_plus_y - y
    assert difference.simplify2() == x.simplify2()

def test_linear_addition_of_binary_expansion():
    v = [BoolVar(f'v_{i}') for i in range(4)]

    x = Linear.of(BinaryExpansion([v[0], 0, v[2]]))
    y = Linear.of(BinaryExpansion([0, v[0], 0, v[2]]))
    result = 5*y
    addition_test(6 * x, 2 * y, result)


    y = Linear.of(BinaryExpansion([v[1], 1, v[3]]))

    x = 3*x + 1
    y = 6*y + 5
    result = Linear.of(BinaryExpansion([v[0], v[1], v[2], v[3]]))
    result = 3 * result + 18
    addition_test(x, y, result)

def test_me():
    vv = [BoolVar(f'v_{i}') for i in range(4)]
    x = BinaryExpansion(vv)
    xx = Linear.of(x)

    cond1 = xx - 7 >= 0
    cond2 = xx - 8 >= 0
    assert cond1 * cond2 == cond2

    cond1 = xx - 7 >= 0
    cond2 = xx - 8 < 0
    prod = cond1 * cond2
    prod = prod.simplify()
    assert prod == SingleAssignmentCondition.from_dict({vv[0]:1, vv[1]:1, vv[2]: 1, vv[3]:0})

    cond1 = xx - 7 < 0
    cond2 = xx - 8 >= 0
    prod = cond1 * cond2
    prod = prod.simplify()
    assert prod == FalseCondition

    cond1 = xx - 7 < 0
    cond2 = xx - 8 < 0
    prod = cond1 * cond2
    prod = prod.simplify()
    assert prod == cond1


def test_assignment_range_condition():
    vv = [BoolVar(f'x_{i}') for i in range(4)]
    x = BinaryExpansion(vv)
    xx = Linear.of(x)

    y = BinaryExpansion(vv[:3])
    yy = Linear.of(y)

    cond1 = (xx-7<0).simplify()
    cond2 = (yy-7<0).simplify()
    cond2 = cond2 & SingleAssignmentCondition(vv[3], 0)
    assert cond1 == cond2

def test_linear_ranged_condition_subtraction():
    vv = [BoolVar(f'x_{i}') for i in range(4)]
    x = BinaryExpansion(vv)
    xx = Linear.of(x)

    v1 = ReLU(xx-7)
    v2 = ReLU(xx-8)
    v = v1 - v2
    v = v.simplify2()

    # TODO: improve union \ intersection of conditions

    # assert v == x.coefficients[3]
    #
    # v = 8 * v
    # u = ConditionalFunction.always(xx) - v
    # u = u.simplify2()
    #
    # result = BinaryExpansion(vv[:3])
    # assert u == result
