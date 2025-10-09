from typing import List

from mapping_field.conditions import FalseCondition
from mapping_field.ranged_condition import RangeCondition, SingleAssignmentCondition
from mapping_field.mapping_field import MapElement, Var


class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([])
        self.value = value

    def to_string(self, vars_str_list: List[str]):
        return f'DummyMap({self.value})'


def test_range_condition_intersection():
    dummy_map = DummyMap(0)

    cond1 = RangeCondition(dummy_map, (0,10))
    cond2 = RangeCondition(dummy_map, (5,15))
    cond12 = RangeCondition(dummy_map, (5,10))
    assert cond1 * cond2 == cond12

    cond3 = RangeCondition(dummy_map, [15,25])
    assert cond1 * cond3 == FalseCondition

def test_range_condition_union():
    dummy_var = Var('x')

    cond1 = RangeCondition(dummy_var, (0,10))
    cond2 = RangeCondition(dummy_var, (5,15))
    cond12 = RangeCondition(dummy_var, (0,15))
    assert cond1 | cond2 == cond12

    cond2 = SingleAssignmentCondition(dummy_var, 10)
    cond12 = RangeCondition(dummy_var, (0,11))
    assert cond1 | cond2 == cond12

    cond3 = RangeCondition(dummy_var, [15,25])
    assert cond1.or_simpler(cond3)[1] == False


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