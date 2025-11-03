import pytest
from typing import List

from mapping_field.binary_expansion import BoolVar, BinaryExpansion
from mapping_field.new_conditions import FalseCondition, TrueCondition
from mapping_field.mapping_field import MapElement, Var, NamedFunc, MapElementConstant
from mapping_field.new_ranged_condition import RangeCondition


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
#
# def test_range_condition_union():
#     dummy_var = Var('x')
#
#     cond1 = RangeCondition(dummy_var, (0,10))
#     cond2 = RangeCondition(dummy_var, (5,15))
#     cond12 = RangeCondition(dummy_var, (0,15))
#     assert cond1 | cond2 == cond12
#
#     cond2 = SingleAssignmentCondition(dummy_var, 10)
#     cond12 = RangeCondition(dummy_var, (0,11))
#     assert cond1 | cond2 == cond12
#
#     cond3 = RangeCondition(dummy_var, [15,25])
#     assert cond1.or_simpler(cond3)[1] == False
#
#
#
#
# def test_extend_range_to_full():
#     vv = [BoolVar(f'x_{i}') for i in range(4)]
#     x = BinaryExpansion(vv)
#
#     cond1 = (x < 16)
#     assert cond1 == TrueCondition
#
#     def from_mid(k: int):
#         cond1 = (x < k)
#         cond2 = (k <= x)
#         cond3 = RangeCondition(x, (k,16))
#         assert cond1 | cond2 == TrueCondition
#         assert cond2 | cond1 == TrueCondition
#         assert cond1 | cond3 == TrueCondition
#         assert cond3 | cond1 == TrueCondition
#
#     from_mid(15)
#     from_mid(9)
#     from_mid(8)
#     from_mid(1)
#     from_mid(0)
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
#
#
# def test_simplifier():
#     x = Var('x')
#     cond1 = (0 <= x - 1)
#     cond1 = cond1.simplify()
#     cond2 = (1 <= x)
#
#     assert cond1 == cond2
#
#
# def test_invert_range():
#     dummy = DummyMap(0)
#
#     condition1 = ~(dummy<3)
#     condition2 = dummy>=3
#     assert condition1 == condition2
#
#     condition1 = ~(dummy<=3)
#     condition2 = dummy>3
#     assert condition1 == condition2
#
#     condition1 = ~ RangeCondition(dummy, (1,3))
#     condition2 = (dummy<1) | (3<=dummy)
#     assert condition1 == condition2
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
