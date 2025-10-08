from typing import List, Tuple

from mapping_field.conditions import (
    Condition, TrueCondition, FalseCondition, ConditionIntersection, ConditionalFunction)
from mapping_field.mapping_field import MapElementConstant, MapElement

class DummyCondition(Condition):
    def __init__(self, value=0):
        super().__init__([])
        self.value = value

    def __repr__(self):
        return f'DummyCond({self.value})'

    def __and__(self, condition: Condition) -> Tuple['Condition', bool]:
        if isinstance(condition, DummyCondition):
            return (self if self.value == condition.value else FalseCondition), True
        return super().__and__(condition)

    def _eq_simplified(self, other: Condition) -> bool:
        return isinstance(other, DummyCondition) and self.value == other.value

class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([], f'DummyMap({self.value})')
        self.value = value


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


def test_intersection_of_unions():
    dummies = [DummyCondition(i) for i in range(5)]

    cond1 = (dummies[0] | dummies[1])[0]
    assert (cond1 * dummies[0]) == dummies[0]
    assert (cond1 * dummies[1]) == dummies[1]
    assert dummies[0] == (cond1 * dummies[0])
    assert dummies[1] == (cond1 * dummies[1])

    cond2 = (dummies[0] | dummies[2])[0]
    assert (cond1 * cond2) == dummies[0]


# Test conditional functions

def test_op_conditional_functions():
    dummies = [DummyCondition(i) for i in range(5)]

    cond_func1 = ConditionalFunction([
        ((dummies[0] | dummies[1])[0], MapElementConstant(0)),
        (dummies[2], MapElementConstant(10))
    ])

    cond_func2 = ConditionalFunction([
        (dummies[0], MapElementConstant(100)),
        ((dummies[1] | dummies[2])[0], MapElementConstant(200))
    ])

    cond_add = cond_func1 + cond_func2

    result = ConditionalFunction([
        (dummies[0], MapElementConstant(100)),
        (dummies[1], MapElementConstant(200)),
        (dummies[2], MapElementConstant(210))
    ])

    assert result == cond_add, f'could not match:\n{result}\n{cond_add}'


