import pytest
from typing import List

from mapping_field.new_conditions import FalseCondition, UnionCondition, TrueCondition
from mapping_field.mapping_field import MapElement, Var, NamedFunc, MapElementConstant
from mapping_field.new_ranged_condition import RangeCondition, InRange
from mapping_field.arithmetics import Add


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var.clear_vars()
    NamedFunc.clear_vars()

class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([])
        self.value = value

    def to_string(self, vars_str_list: List[str]):
        return f'DummyMap({self.value})'


def test_comparison_operators():
    dummy = DummyMap(0)

    cond = (dummy < 10)
    assert cond == RangeCondition(dummy, (float('-inf'), 10))

    cond = (dummy <= 10)
    assert cond == RangeCondition(dummy, (float('-inf'), 11))

    cond = (10 <= dummy)
    assert cond == RangeCondition(dummy, (10, float('inf')))

    cond = (10 < dummy)
    assert cond == RangeCondition(dummy, (11, float('inf')))

    # Unfortunately, python is terrible, and I can't use 2-sided comparisons like:
    #   cond = (10 <= dummy < 20)

def test_invert_range():
    dummy = DummyMap(0)

    condition1 = ~(dummy<3)
    condition2 = dummy>=3
    assert condition1 == condition2

    condition1 = ~(dummy<=3)
    condition2 = dummy>3
    assert condition1 == condition2

    condition1 = ~ RangeCondition(dummy, (1,3))
    condition2 = (dummy<1) | (3<=dummy)
    assert condition1 == condition2

def test_range_condition_intersection():
    dummy_map = DummyMap(0)

    # closed ranges
    cond1 = RangeCondition(dummy_map, (0,10))
    cond2 = RangeCondition(dummy_map, (5,15))
    cond12 = RangeCondition(dummy_map, (5,10))
    assert cond1 & cond2 == cond12

    # half open ranges
    cond1 = (5 <= dummy_map)
    cond2 = (dummy_map < 10)
    cond12 = RangeCondition(dummy_map, (5,10))
    assert cond1 & cond2 == cond12

    # disjoint ranges
    cond1 = RangeCondition(dummy_map, (0,10))
    cond2 = RangeCondition(dummy_map, (15,25))
    assert cond1 & cond2 == FalseCondition

def test_range_condition_union():
    dummy_map = DummyMap(0)

    # Union with intersection
    cond1 = RangeCondition(dummy_map, (0,10))
    cond2 = RangeCondition(dummy_map, (5,15))
    cond12 = RangeCondition(dummy_map, (0,15))
    assert cond1 | cond2 == cond12

    # Consecutive ranges with no intersection
    cond1 = RangeCondition(dummy_map, (0,8))
    cond2 = RangeCondition(dummy_map, (8,15))
    cond12 = RangeCondition(dummy_map, (0,15))
    assert cond1 | cond2 == cond12

    # Purely disjoint ranges
    cond1 = RangeCondition(dummy_map, (0,5))
    cond2 = RangeCondition(dummy_map, (10,15))
    cond12 = UnionCondition([cond1, cond2])
    assert cond12._simplify2() is None

    # cond2 = SingleAssignmentCondition(dummy_var, 10)
    # cond12 = RangeCondition(dummy_var, (0,11))
    # assert cond1 | cond2 == cond12

def test_simplify_all_or_nothing_range():
    dummy_map = DummyMap(0)

    cond = RangeCondition(dummy_map, (10,10))
    assert cond is not FalseCondition
    assert cond.simplify2() is FalseCondition

    cond = RangeCondition(dummy_map, (20,10))
    assert cond is not FalseCondition
    assert cond.simplify2() is FalseCondition

    cond = RangeCondition(dummy_map, (float('-inf'),float('+inf')))
    assert cond is not TrueCondition
    assert cond.simplify2() is TrueCondition

def test_simplify_evaluated():
    c = Add(MapElementConstant(3), MapElementConstant(2), simplify=False)
    assert str(c) != str(MapElementConstant(5))
    # c is not by itself a constant, but can be evaluated into a constant

    assert (c < 10).simplify2() == TrueCondition
    assert (1 <= c).simplify2() == TrueCondition
    assert (RangeCondition(c, (1,10))).simplify2() == TrueCondition

    assert (c > 10).simplify2() == FalseCondition
    assert (1 >= c).simplify2() == FalseCondition
    assert (RangeCondition(c, (10,100))).simplify2() == FalseCondition

def test_simplify_linear_ranged_condition():
    dummy = DummyMap(0)

    def inner_test(a, b, low, high) -> None:
        cond1 = RangeCondition(dummy, (low, high))
        low, high = a*low + b, a*high + b
        if a<0:
            low, high = high+1, low+1
        cond2 = RangeCondition(a*dummy+b, (low, high))
        cond2 = cond2.simplify2()
        assert cond1 == cond2

    inner_test(a=2, b=3, low=1, high=6)
    inner_test(a=-2, b=3, low=-5, high=0)

    # test zero coefficient - False
    condition = RangeCondition(0*dummy+3, (5, 15))
    condition = condition.simplify2()
    assert condition == FalseCondition

    # test zero coefficient - False
    condition = RangeCondition(0*dummy+7, (5, 15))
    condition = condition.simplify2()
    assert condition == TrueCondition

def test_simplify_on_ranged_promised_functions():
    dummy = DummyMap(0)

    # Simple condition, can't be simplified.
    condition = (5 <= dummy)
    condition = condition._simplify2()
    assert condition is None

    # Add to dummy a nonnegative output assumption
    dummy.add_promise( InRange((0, float('inf'))) )

    condition = (-5 <= dummy)
    condition = condition.simplify2()
    assert condition is TrueCondition

    condition = (dummy < -5)
    condition = condition.simplify2()
    assert condition is FalseCondition

    # new condition is contained inside the assumption, so no change
    condition = (5 <= dummy)
    condition = condition._simplify2()
    assert condition is None

    # intersection of new condition and assumption
    condition = (dummy < 5)
    condition = condition.simplify2()
    result = RangeCondition(dummy, (0, 5))
    assert condition is not result



#
#
#
#

#
# def test_general_assignment():
#     dummy = DummyMap(0)
#
#     assert (dummy.where() == 3) == GeneralAssignment(dummy, 3)
#
#     assert (MapElementConstant(3).where() == 3) is TrueCondition
#
#     assert (MapElementConstant(5).where() == 3) is FalseCondition
#
#
# def test_general_range_condition_union():
#     dummy = DummyMap(0)
#
#     # contained:
#     cond1 = (dummy.where() == 10)
#     cond2 = RangeCondition(dummy, (0,11))
#     assert cond1 | cond2 == cond2
#
#     # union of two assignments
#     cond1 = (dummy.where() == 10)
#     cond2 = (dummy.where() == 11)
#     cond3 = RangeCondition(dummy, (10,12))
#     assert cond1 | cond2 == cond3
#
#     # union of assignment and range
#     cond1 = (dummy.where() == 10)
#     cond2 = RangeCondition(dummy, (5,10))
#     cond3 = RangeCondition(dummy, (5,11))
#     assert cond1 | cond2 == cond3
#
#     # disjoint
#     cond1 = (dummy.where() == 10)
#     cond2 = RangeCondition(dummy, [15,25])
#     assert cond1.or_simpler(cond2)[1] == False
#
#
# def test_general_range_condition_intersection():
#     dummy = DummyMap(0)
#
#     # same:
#     cond1 = (dummy.where() == 10)
#     cond2 = (dummy.where() == 10)
#     assert cond1 & cond2 == cond1
#
#     # distinct:
#     cond1 = (dummy.where() == 10)
#     cond2 = (dummy.where() == 13)
#     assert cond1 & cond2 == FalseCondition
#
#     # contained
#     cond1 = (dummy.where() == 10)
#     cond2 = RangeCondition(dummy, (5,12))
#     assert cond1 & cond2 == cond1
#
#     # distinct
#     cond1 = (dummy.where() == 10)
#     cond2 = RangeCondition(dummy, (12,17))
#     assert cond1 & cond2 == FalseCondition
