from typing import List, Tuple, Union, Set

from mapping_field.new_conditions import TrueCondition, FalseCondition, IsCondition, IntersectionCondition
from mapping_field.mapping_field import MapElement

class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([], f'DummyMap({self.value})')
        self.value = value

class DummyCondition(MapElement):
    def __init__(self, values: Union[int, Set[int]]=0, type: int=0):
        super().__init__([])
        self.values: Set[int] = set([values]) if isinstance(values, int) else values
        self.type = type
        self.add_promise(IsCondition)

    def to_string(self, vars_str_list: List[str]):
        return f'DummyCond_{self.type}({self.values})'


def test_binary_condition_invert():
    assert ~TrueCondition == FalseCondition
    assert ~FalseCondition == TrueCondition

    dummy = DummyCondition()
    not_dummy = ~dummy
    not_not_dummy = ~not_dummy

    assert dummy == not_not_dummy

def test_binary_conditions_and():
    dummy = DummyCondition(0)

    assert dummy & TrueCondition == dummy
    assert TrueCondition & dummy == dummy
    assert dummy & FalseCondition == FalseCondition
    assert FalseCondition & dummy == FalseCondition
    assert dummy & dummy == dummy

def test_binary_conditions_or():
    dummy = DummyCondition(0)

    assert dummy | TrueCondition == TrueCondition
    assert TrueCondition | dummy == TrueCondition
    assert dummy | FalseCondition == dummy
    assert FalseCondition | dummy == dummy
#
# def test_unpack_intersections():
#     dummies = [DummyCondition(type = i) for i in range(5)]
#
#     cond1 = ConditionIntersection([dummies[0], dummies[1]])
#
#     cond2 = ConditionIntersection([cond1, dummies[2]])
#     assert isinstance(cond2, ConditionIntersection) and len(cond2.conditions) == 3
#
#     cond2 = ConditionIntersection([dummies[2], cond1])
#     assert isinstance(cond2, ConditionIntersection) and len(cond2.conditions) == 3
#
# def test_simplify_intersection():
#     dummies = [DummyCondition(i) for i in range(5)] # TODO: type = i ?
#
#     cond1 = ConditionIntersection([dummies[0], dummies[1], dummies[2]])
#     cond2 = ConditionIntersection([dummies[2], dummies[0], dummies[1]])
#     assert cond1 == cond2
#
#     cond1 = ConditionIntersection([TrueCondition, dummies[0], dummies[1], dummies[2], TrueCondition])
#     cond2 = ConditionIntersection([dummies[2], dummies[0], dummies[1]])
#     assert cond1 == cond2
#
#     cond1 = ConditionIntersection([FalseCondition, dummies[0], dummies[1], dummies[2], TrueCondition])
#     cond2 = FalseCondition
#     assert cond1 == cond2
#
#     cond1 = ConditionIntersection([dummies[0], dummies[1], dummies[1], dummies[0], dummies[2]])
#     cond2 = ConditionIntersection([dummies[2], dummies[0], dummies[1]])
#     assert cond1 == cond2
#
#     cond1 = ConditionIntersection([dummies[0]])
#     cond2 = dummies[0]
#     assert cond1 == cond2
    assert dummy | dummy == dummy

def test_binary_and_with_invert():
    dummy0 = DummyCondition(0)
    dummy1 = DummyCondition(1)

    assert dummy0 & ~dummy0 == FalseCondition
    assert ~dummy0 & dummy0 == FalseCondition
    assert str((~dummy0) & (~dummy1)) == str(~(dummy0 | dummy1))

def test_binary_or_with_invert():
    dummy0 = DummyCondition(0)
    dummy1 = DummyCondition(1)

    assert dummy0 | ~dummy0 == TrueCondition
    assert ~dummy0 | dummy0 == TrueCondition
    assert str((~dummy0) | (~dummy1)) == str(~(dummy0 & dummy1))
#
# def test_improved_simplify_intersection():
#     dummies = [DummyCondition(i) for i in range(5)]
#
#     assert dummies[0] & dummies[0] == dummies[0]
#
#     cond1 = ConditionIntersection([dummies[0], dummies[1], dummies[2]])
#     cond2 = ConditionIntersection([dummies[0], dummies[1], dummies[3]])
#     prod = cond1 & cond2
#     result = ConditionIntersection([dummies[0], dummies[1], dummies[2], dummies[3]])
#     assert prod == result
#
#     prod = (dummies[0] & dummies[1] & dummies[2]) & (dummies[0] & dummies[1] & dummies[3])
#     result = (dummies[0] & dummies[1] & dummies[2] & dummies[3])
#     assert prod == result
#
# def test_union_of_intersections():
#     dummies = [DummyCondition(values=0, type=i) for i in range(5)]
#
#     # containment:
#
#     cond1 = dummies[0] & dummies[1] & dummies[2]
#     cond2 = dummies[2] & dummies[0]
#     cond3 = dummies[2]
#
#     assert cond1 | cond2 == cond2
#     assert cond2 | cond1 == cond2
#
#     assert cond1 | cond3 == cond3
#     assert cond3 | cond1 == cond3
#
#     assert cond2 | cond3 == cond3
#     assert cond3 | cond2 == cond3
#
#     # One Union
#
#     cond1 = dummies[0] & dummies[1] & dummies[2]
#     dummy_special = DummyCondition(values = 1, type = 0)
#     cond2 = dummy_special & dummies[1] & dummies[2]
#     dummy_union = DummyCondition(values = {0,1}, type = 0)
#
#     result = dummies[1] & dummy_union & dummies[2]
#     assert cond1 | cond2 == result
#
# def test_intersection_of_unions():
#     dummies = [DummyCondition(i) for i in range(5)]
#
#     # containment:
#     cond1 = dummies[0] | dummies[1] | dummies[2]
#     cond2 = dummies[2] | dummies[0]
#     cond3 = dummies[2]
#
#     assert cond1 & cond2 == cond2
#     assert cond2 & cond1 == cond2
#
#     assert cond1 & cond3 == cond3
#     assert cond3 & cond1 == cond3
#
#     assert cond2 & cond3 == cond3
#     assert cond3 & cond2 == cond3
#
#     # One intersection
#
#     dummy01 = DummyCondition(values={0,1}, type = 0)
#     dummy02 = DummyCondition(values={0,2}, type = 0)
#     dummy0  = DummyCondition(values={0}, type = 0)
#
#     cond1 = dummy01 | dummies[1] | dummies[2]
#     cond2 = dummy02 | dummies[1] | dummies[2]
#     result = dummy0 | dummies[1] | dummies[2]
#
#     assert cond1 & cond2 == result
#
# def single_containment(cond_small: Condition, cond_large: Condition):
#
#     # TODO: Make sure that the reverse works as well
#     intersection = cond_small & cond_large
#     assert intersection == cond_small
#     intersection = cond_large & cond_small
#     assert intersection == cond_small
#
# def test_containment_in_union():
#
#     dummies1 = [DummyCondition(values={1},type = i) for i in range(5)]
#     dummies3 = [DummyCondition(values={1,2,3},type = i) for i in range(5)]
#
#     condition = dummies3[0] | dummies3[1] | dummies3[2] | dummies3[3]
#
#     # Exactly one of the element of the union
#     single_containment(dummies3[1], condition)
#
#     # Contained in one of the elements in the union
#     single_containment(dummies1[1], dummies3[1])  # make sure that dummies1 is inside dummies3 first
#     single_containment(dummies1[1], condition)
#
#     # Exactly two elements the same
#     single_containment(dummies3[1] | dummies3[3], condition)
#
#     # One element the same, and another contained
#     single_containment(dummies3[1] | dummies1[3], condition)
#
# def test_containment_in_intersection():
#
#     dummies1 = [DummyCondition(values={1},type = i) for i in range(5)]
#     dummies3 = [DummyCondition(values={1,2,3},type = i) for i in range(5)]
#
#     condition = dummies1[0] & dummies1[1] & dummies1[2] & dummies1[3]
#
#     # Exactly one of the element of the intersection
#     single_containment(condition, dummies1[1])
#
#     # Containing one of the elements from the intersection
#     single_containment(dummies1[1], dummies3[1])  # make sure that dummies1 is inside dummies3 first
#     single_containment(condition, dummies3[1])
#
#     # Exactly two elements the same
#     single_containment(condition, dummies1[1] & dummies1[3])
#
#     # One element the same, and another contained
#     single_containment(condition, dummies3[1] & dummies1[3])
#
# def test_simplified_intersection_list():
#
#     dummies12 = [DummyCondition(values={1,2}, type = i) for i in range(5)]
#     dummies13 = [DummyCondition(values={1,3}, type = i) for i in range(5)]
#     dummies1  = [DummyCondition(values={1  }, type = i) for i in range(5)]
#
#     # condition = dummies12[0] | dummies12[1] | dummies12[2] | dummies12[3]
#     #
#     # # Exactly one of the element of the intersection
#     condition = (dummies12[0] | dummies1[1] | dummies12[2] | dummies12[3]) & dummies13[1]
#     result = condition & dummies13[1]
#     print('\n^^^^^^^^^^^^^^^^^^^')
#     print(condition)
#     print(result)
#     assert str(condition) == str(result)
#
#
# def test_intersection_of_union_component_simplification():
#     # If A1 and A2 has some nontrivial intersection, then
#     #   (A1 | B | C) & A2 = (A1 & A2) | (B & A2) | (C & A2)
#     #                     = (A1 & A2 & A2) | (B & A2) | (C & A2)
#     #                     = ((A1 & A2) | B | C ) & A2
#     # In particular, if A1 and A2 do not intersect, we are left with
#     #                     = (B | C) & A2
#
#     dummies1 = [DummyCondition(values={1},type = i) for i in range(5)]
#     dummies3 = [DummyCondition(values={1,2,3},type = i) for i in range(5)]
#
#     condition1 = dummies3[0] | dummies3[1] | dummies3[2] | dummies3[3]
#     condition2 = dummies3[0] | dummies3[1]
#
#     # assert condition1 & condition2 == condition2
#
#     # small_dummy is contained in each of the dummies3[i]
#     small_dummy = dummies1[0] & dummies1[1] & dummies1[2] & dummies1[3]
#
#     intersection = small_dummy & dummies3[0]
#     assert intersection == small_dummy
#
#     union = small_dummy | dummies3[0]
#     assert union == dummies3[0]
#
#     intersection = small_dummy & ( dummies3[0] | dummies3[1] )
#     assert intersection == small_dummy
#
#     intersection = (dummies3[0] | small_dummy) & ( dummies3[0] | dummies3[1] )
#     assert intersection == (dummies3[0] | small_dummy)
#     #
#     # assert intersection == small_dummy
#