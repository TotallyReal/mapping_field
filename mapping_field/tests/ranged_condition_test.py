from typing import List

from mapping_field.conditions import FalseCondition, ConditionalFunction
from mapping_field.ranged_condition import RangeCondition, AssignmentCondition
from mapping_field.mapping_field import MapElementConstant, MapElement


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
    dummy_map = DummyMap(0)

    cond1 = RangeCondition(dummy_map, (0,10))
    cond2 = RangeCondition(dummy_map, (5,15))
    cond12 = RangeCondition(dummy_map, (0,15))
    assert (cond1 | cond2)[0] == cond12

    cond2 = AssignmentCondition({dummy_map: 10})
    cond12 = RangeCondition(dummy_map, (0,11))
    assert (cond1 | cond2)[0] == cond12

    cond3 = RangeCondition(dummy_map, [15,25])
    assert (cond1 | cond3)[1] == False

# Test conditional functions

def test_op_conditional_functions_ranges():
    dummy_map = DummyMap(0)

    def ranged(low, high):
        return RangeCondition(dummy_map, (low, high))

    cond_func1 = ConditionalFunction([
        (ranged(0,10), MapElementConstant(0)),
        (ranged(10,30), MapElementConstant(10))
    ])

    cond_func2 = ConditionalFunction([
        (ranged(0,20), MapElementConstant(100)),
        (ranged(20,30), MapElementConstant(200))
    ])

    cond_add = cond_func1 + cond_func2

    result = ConditionalFunction([
        (ranged(10,20), MapElementConstant(110)),
        (ranged(0,10), MapElementConstant(100)),
        (ranged(20,30), MapElementConstant(210))
    ])

    assert  result == cond_add