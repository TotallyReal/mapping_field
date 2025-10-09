from typing import List, Tuple, Union, Set

from mapping_field.conditions import Condition, TrueCondition, FalseCondition, ConditionIntersection
from mapping_field.mapping_field import MapElement

class DummyCondition(Condition):
    def __init__(self, values: Union[int, Set[int]]=0, type: int=0):
        super().__init__([])
        self.values: Set[int] = set([values]) if isinstance(values, int) else values
        self.type = type

    def __repr__(self):
        return f'DummyCond_{self.type}({self.values})'

    def and_simpler(self, condition: Condition) -> Tuple['Condition', bool]:
        if isinstance(condition, DummyCondition) and self.type == condition.type:
            intersection = self.values.intersection(condition.values)
            return DummyCondition(intersection) if len(intersection) > 0 else FalseCondition, True
        return super().and_simpler(condition)

    def or_simpler(self, condition: Condition) -> Tuple['Condition', bool]:
        if isinstance(condition, DummyCondition) and self.type == condition.type:
            union = self.values.union(condition.values)
            return DummyCondition(union), True
        return super().or_simpler(condition)

    def _eq_simplified(self, other: Condition) -> bool:
        return (isinstance(other, DummyCondition) and
                self.type == other.type and
                len(self.values) == len(other.values) and
                all([v in other.values for v in self.values]))

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

def test_union_of_intersections():
    dummies = [DummyCondition(values=0, type=i) for i in range(5)]

    # containment:

    cond1 = dummies[0] * dummies[1] * dummies[2]
    cond2 = dummies[2] * dummies[0]
    cond3 = dummies[2]

    assert cond1 | cond2 == cond2
    assert cond2 | cond1 == cond2

    assert cond1 | cond3 == cond3
    assert cond3 | cond1 == cond3

    assert cond2 | cond3 == cond3
    assert cond3 | cond2 == cond3

    # One Union

    cond1 = dummies[0] * dummies[1] * dummies[2]
    dummy_special = DummyCondition(values = 1, type = 0)
    cond2 = dummy_special * dummies[1] * dummies[2]
    dummy_union = DummyCondition(values = {0,1}, type = 0)

    result = dummies[1] * dummy_union * dummies[2]
    assert cond1 | cond2 == result

def test_intersection_of_unions():
    dummies = [DummyCondition(i) for i in range(5)]

    # containment:
    cond1 = dummies[0] | dummies[1] | dummies[2]
    cond2 = dummies[2] | dummies[0]
    cond3 = dummies[2]

    assert cond1 * cond2 == cond2
    assert cond2 * cond1 == cond2

    assert cond1 * cond3 == cond3
    assert cond3 * cond1 == cond3

    assert cond2 * cond3 == cond3
    assert cond3 * cond2 == cond3

    # One intersection

    dummy01 = DummyCondition(values={0,1}, type = 0)
    dummy02 = DummyCondition(values={0,2}, type = 0)
    dummy0  = DummyCondition(values={0}, type = 0)

    cond1 = dummy01 | dummies[1] | dummies[2]
    cond2 = dummy02 | dummies[1] | dummies[2]
    result = dummy0 | dummies[1] | dummies[2]

    assert cond1 * cond2 == result

