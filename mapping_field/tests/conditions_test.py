from typing import List

from mapping_field.conditions import (
    Condition, TrueCondition, FalseCondition, RangeCondition, ConditionIntersection, ConditionalFunction, ReLU,
    AssignmentCondition)
from mapping_field.mapping_field import MapElementConstant, MapElement

class DummyCondition(Condition):
    def __init__(self, value=0):
        super().__init__([])
        self.value = value

    def __repr__(self):
        return f'DummyCond({self.value})'

class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([])
        self.value = value

    def to_string(self, vars_str_list: List[str]):
        return f'DummyMap({self.value})'


# Test conditions

def test_binary_conditions():
    dummy = DummyCondition(0)
    assert dummy * TrueCondition == dummy
    assert TrueCondition * dummy == dummy
    assert dummy * FalseCondition == FalseCondition
    assert FalseCondition * dummy == FalseCondition

def test_simplify_intersection():
    dummies = [DummyCondition(i) for i in range(5)]

    cond1 = ConditionIntersection([dummies[0], dummies[1], dummies[2]])
    cond2 = ConditionIntersection([dummies[2], dummies[0], dummies[1]])
    assert cond1 == cond2

    cond1 = ConditionIntersection([TrueCondition, dummies[0], dummies[1], dummies[2], TrueCondition])
    cond2 = ConditionIntersection([dummies[2], dummies[0], dummies[1]])
    assert cond1 == cond2

    cond1 = ConditionIntersection([FalseCondition, dummies[0], dummies[1], dummies[2], TrueCondition])
    cond2 = FalseCondition
    assert cond1 == cond2

    cond1 = ConditionIntersection([dummies[0], dummies[1], dummies[1], dummies[0], dummies[2]])
    cond2 = ConditionIntersection([dummies[2], dummies[0], dummies[1]])
    assert cond1 == cond2

    cond1 = ConditionIntersection([dummies[0]])
    cond2 = dummies[0]
    assert cond1 == cond2

def test_improved_simplify_intersection():
    dummies = [DummyCondition(i) for i in range(5)]

    assert dummies[0] * dummies[0] == dummies[0]

    cond1 = ConditionIntersection([dummies[0], dummies[1], dummies[2]])
    cond2 = ConditionIntersection([dummies[0], dummies[1], dummies[3]])
    prod = cond1 * cond2
    result = ConditionIntersection([dummies[0], dummies[1], dummies[2], dummies[3]])
    assert prod == result

    prod = (dummies[0] * dummies[1] * dummies[2]) * (dummies[0] * dummies[1] * dummies[3])
    result = (dummies[0] * dummies[1] * dummies[2] * dummies[3])
    assert prod == result

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

def test_op_conditional_functions():
    dummies = [DummyCondition(i) for i in range(5)]

    cond_func1 = ConditionalFunction([
        (dummies[0], MapElementConstant(0)),
        (dummies[1], MapElementConstant(10))
    ])

    cond_func2 = ConditionalFunction([
        (dummies[0], MapElementConstant(100)),
        (dummies[2], MapElementConstant(200))
    ])

    cond_add = cond_func1 + cond_func2

    result = ConditionalFunction([
        (dummies[0], MapElementConstant(100)),
        (dummies[0] * dummies[2], MapElementConstant(200)),
        (dummies[1] * dummies[2], MapElementConstant(210)),
        (dummies[1] * dummies[0], MapElementConstant(110))
    ])

    assert result == cond_add, f'could not match:\n{result}\n{cond_add}'

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
