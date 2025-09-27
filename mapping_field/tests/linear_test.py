import pytest
from typing import List

from mapping_field.linear import Linear, IntVar
from mapping_field.conditions import RangeCondition, AssignmentCondition, FalseCondition, TrueCondition, Condition
from mapping_field.mapping_field import MapElementConstant, MapElement, Var

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
    func = func.simplify()
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
    cond1 = AssignmentCondition({x:1, y:2})
    cond2 = AssignmentCondition({z:3})
    prod = cond1 * cond2
    result = AssignmentCondition({x:1, y:2, z:3})
    assert prod == result

    # valid intersection
    cond1 = AssignmentCondition({x:1, y:2})
    cond2 = AssignmentCondition({y:2, z:3})
    prod = cond1 * cond2
    result = AssignmentCondition({x:1, y:2, z:3})
    assert prod == result

    # invalid intersection
    cond1 = AssignmentCondition({x:1, y:2})
    cond2 = AssignmentCondition({y:5, z:3})
    prod = cond1 * cond2
    result = FalseCondition
    assert prod == result


def test_assignment_from_range():
    x = IntVar('x')

    condition = RangeCondition(x, (5,6))
    condition = condition.simplify()
    result = AssignmentCondition({x: 5})
    assert condition == result

    linear_func = Linear(5, x, 2)
    condition = RangeCondition(linear_func, (7,12))
    condition = condition.simplify()
    result = AssignmentCondition({x: 1})
    assert condition == result

