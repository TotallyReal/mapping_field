from typing import List, Tuple, Union, Set, Type, Optional

import pytest

from mapping_field.new_conditions import TrueCondition, FalseCondition, IsCondition, IntersectionCondition, \
    UnionCondition, _ListCondition
from mapping_field.mapping_field import MapElement

class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([], f'DummyMap({self.value})')
        self.value = value

class DummyCondition(MapElement):
    def __init__(self, type: int=0, values: Union[int, Set[int]]=0):
        super().__init__([])
        self.values: Set[int] = set([values]) if isinstance(values, int) else values
        self.type = type
        self.add_promise(IsCondition)

    def to_string(self, vars_str_list: List[str]):
        return f'DummyCond_{self.type}({self.values})'

    def and_(self, condition: MapElement) -> Optional[MapElement]:
        if isinstance(condition, DummyCondition) and self.type == condition.type:
            intersection = self.values.intersection(condition.values)
            return DummyCondition(values=intersection, type=self.type) if len(intersection) > 0 else FalseCondition
        return None

    def or_(self, condition: MapElement) -> Optional[MapElement]:
        if isinstance(condition, DummyCondition) and self.type == condition.type:
            union = self.values.union(condition.values)
            return DummyCondition(values=union, type=self.type)
        return None

    def __eq__(self, other: MapElement) -> bool:
        return (isinstance(other, DummyCondition) and
                self.type == other.type and
                len(self.values) == len(other.values) and
                all([v in other.values for v in self.values]))

#       ╭─────────────────────────────────────────────────╮
#       │           Binary And \ Or \ Invert              │
#       ╰─────────────────────────────────────────────────╯

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

#       ╭─────────────────────────────────────────────────╮
#       │                 List Conditions                 │
#       ╰─────────────────────────────────────────────────╯

@pytest.fixture(params=["Intersection", "Union"])
def dual_case(request):
    op_name = request.param

    dummies1 = [DummyCondition(values={1},type = i) for i in range(5)]
    dummies2 = [DummyCondition(values={1,2},type = i) for i in range(5)]
    if op_name == 'Intersection':
        return IntersectionCondition, dummies1, dummies2
    if op_name == 'Union':
        return UnionCondition, dummies2, dummies1
    raise NotImplementedError('How did you even get here?!')

# With the following fixtures we have that:
# list_class([weak_dummy, strong_dummy]) = strong_dummy

@pytest.fixture
def list_class(dual_case) -> Type[_ListCondition]:
    return dual_case[0]

@pytest.fixture
def strong_dummies(dual_case):
    return dual_case[1]

@pytest.fixture
def weak_dummies(dual_case):
    return dual_case[2]


def test_intersection_with_delim():
    dummies = [DummyCondition(type = i) for i in range(4)]

    cond1 = IntersectionCondition([dummies[0], dummies[1], dummies[2], dummies[3]])
    cond2 = dummies[0] & dummies[1] & dummies[2] & dummies[3]
    assert cond1 == cond2

def test_union_with_delim():
    dummies = [DummyCondition(type = i) for i in range(4)]

    cond1 = UnionCondition([dummies[0], dummies[1], dummies[2], dummies[3]])
    cond2 = dummies[0] | dummies[1] | dummies[2] | dummies[3]
    assert cond1 == cond2

def test_unpack_lists(list_class: Type[_ListCondition]):
    """
    Note that unpacking is done in the constructor without any simplification.
    """
    dummies = [DummyCondition(type = i) for i in range(5)]

    cond1 = list_class([dummies[0], dummies[1]])

    # unpacking one level
    cond2 = list_class([cond1, dummies[2]])
    assert isinstance(cond2, list_class) and len(cond2.conditions) == 3

    cond2 = list_class([dummies[2], cond1])
    assert isinstance(cond2, list_class) and len(cond2.conditions) == 3

    # un packing two levels
    cond3 = list_class([dummies[3], cond2])
    assert isinstance(cond3, list_class) and len(cond3.conditions) == 4

def test_equality_list_permutations(list_class: Type[_ListCondition]):
    dummies = [DummyCondition(i) for i in range(5)]

    cond1 = list_class([dummies[0], dummies[1], dummies[2]])
    cond2 = list_class([dummies[2], dummies[0], dummies[1]])
    assert cond1 == cond2

# ------------------ Simplifiers ------------------

def test_simplify_trivial_condition(list_class: Type[_ListCondition]):
    dummies = [DummyCondition(type=i) for i in range(5)]

    trivial_condition = list_class.trivials[list_class.type]

    cond1 = list_class([trivial_condition, dummies[0], dummies[1], dummies[2], trivial_condition])
    cond1 = cond1.simplify2()
    cond2 = list_class([dummies[2], dummies[0], dummies[1]])
    assert cond1 == cond2

def test_simplify_final_condition(list_class: Type[_ListCondition]):
    dummies = [DummyCondition(type=i) for i in range(5)]

    trivial_condition = list_class.trivials[list_class.type]
    final_condition = list_class.trivials[1-list_class.type]

    # Controlled by final_condition
    cond1 = list_class([final_condition, dummies[0], dummies[1], dummies[2], trivial_condition])
    cond1 = cond1.simplify2()
    cond2 = final_condition
    assert cond1 == cond2

def test_simplify_repeating_conditions(list_class: Type[_ListCondition]):
    dummies = [DummyCondition(type=i) for i in range(5)]

    cond1 = list_class([dummies[0], dummies[1], dummies[1], dummies[0], dummies[2]])
    cond1 = cond1.simplify2()
    cond2 = list_class([dummies[2], dummies[0], dummies[1]])
    assert cond1 == cond2

def test_simplify_unwrapping(list_class: Type[_ListCondition]):
    dummies = [DummyCondition(type=i) for i in range(5)]

    cond1 = list_class([dummies[0]])
    cond1 = cond1.simplify2()
    cond2 = dummies[0]
    assert cond1 == cond2

def test_simplify_list_of_lists(list_class: Type[_ListCondition]):
    dummies = [DummyCondition(type=i) for i in range(5)]

    cond1 = list_class([
        list_class([dummies[0], dummies[1], dummies[2]]),
        list_class([dummies[0], dummies[1], dummies[3]])
    ])
    cond1 = cond1.simplify2()
    cond2 = list_class([dummies[0], dummies[1], dummies[2], dummies[3]])
    assert cond1 == cond2

    cond1 = list_class([
        list_class([dummies[0], dummies[1], dummies[2]]),
        list_class([dummies[0], dummies[1], dummies[3]]),
        list_class([dummies[4], dummies[2]]),
        list_class([dummies[1], dummies[0]]),
        list_class([dummies[1]]),
        list_class([])
    ])
    cond1 = cond1.simplify2()
    cond2 = list_class([dummies[0], dummies[1], dummies[2], dummies[3], dummies[4]])
    assert cond1 == cond2

def test_simplify_containment(list_class: Type[_ListCondition]):
    # Test cases like   (A & B & C & D) | (A & B) = (A & B)
    #                   (A | B | C | D) & (A | B) = (A | B)
    rev_op = list_class.op_types[1-list_class.type]

    dummies = [DummyCondition(values=0, type=i) for i in range(5)]

    conditions = [
        list_class([dummies[0] , dummies[1] , dummies[2]]),
        list_class([dummies[0] , dummies[2]]),
        dummies[2]
    ]

    for i in range(3):
        for j in range(i):
            assert rev_op(conditions[i], conditions[j]) == conditions[i], f'Failed at indices {i=}, {j=}'

def test_simplification_list_with_special_union_condition():
    dummies = [DummyCondition(values=0, type=i) for i in range(5)]

    cond1 = dummies[0] & dummies[1] & DummyCondition(values = 1, type = 2)
    cond2 = dummies[0] & dummies[1] & DummyCondition(values = 2, type = 2)
    result = dummies[0] & dummies[1] & DummyCondition(values = {1,2}, type = 2)

    assert cond1 | cond2 == result

def test_simplification_list_with_special_intersection_condition():
    dummies = [DummyCondition(values=0, type=i) for i in range(5)]

    cond1 = dummies[0] | dummies[1] | DummyCondition(values = {1,2}, type = 2)
    cond2 = dummies[0] | dummies[1] | DummyCondition(values = {1,3}, type = 2)
    result = dummies[0] | dummies[1] | DummyCondition(values = {1}, type = 2)

    assert cond1 & cond2 == result

def single_containment(cond_small: MapElement, cond_large: MapElement):

    # TODO: Make sure that the reverse works as well
    intersection = cond_small & cond_large
    assert intersection == cond_small
    intersection = cond_large & cond_small
    assert intersection == cond_small

def test_containment_in_union():

    dummies1 = [DummyCondition(values={1},type = i) for i in range(5)]
    dummies3 = [DummyCondition(values={1,2,3},type = i) for i in range(5)]

    condition = dummies3[0] | dummies3[1] | dummies3[2] | dummies3[3]

    # Exactly one of the element of the union
    single_containment(dummies3[1], condition)

    # Contained in one of the elements in the union
    single_containment(dummies1[1], dummies3[1])  # make sure that dummies1 is inside dummies3 first
    single_containment(dummies1[1], condition)

    # Exactly two elements the same
    single_containment(dummies3[1] | dummies3[3], condition)

    # One element the same, and another contained
    single_containment(dummies3[1] | dummies1[3], condition)

def test_containment_in_intersection():

    dummies1 = [DummyCondition(values={1},type = i) for i in range(5)]
    dummies3 = [DummyCondition(values={1,2,3},type = i) for i in range(5)]

    condition = dummies1[0] & dummies1[1] & dummies1[2] & dummies1[3]

    # Exactly one of the element of the intersection
    single_containment(condition, dummies1[1])

    # Containing one of the elements from the intersection
    single_containment(dummies1[1], dummies3[1])  # make sure that dummies1 is inside dummies3 first
    single_containment(condition, dummies3[1])

    # Exactly two elements the same
    single_containment(condition, dummies1[1] & dummies1[3])

    # One element the same, and another contained
    single_containment(condition, dummies3[1] & dummies1[3])

# def test_simplified_intersection_list(simple_logs):
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
#     assert str(condition) == str(result)
#
#
def test_intersection_of_union_component_simplification():
    # If A1 and A2 have some nontrivial intersection, then
    #   (A1 | B | C) & A2 = (A1 & A2) | (B & A2) | (C & A2)
    #                     = (A1 & A2 & A2) | (B & A2) | (C & A2)
    #                     = ((A1 & A2) | B | C ) & A2
    # In particular, if A1 and A2 do not intersect, we are left with
    #                     = (B | C) & A2

    dummies1 = [DummyCondition(values={1},type = i) for i in range(5)]
    dummies3 = [DummyCondition(values={1,2,3},type = i) for i in range(5)]

    condition1 = dummies3[0] | dummies3[1] | dummies3[2] | dummies3[3]
    condition2 = dummies3[0] | dummies3[1]

    assert condition1 & condition2 == condition2

    # small_dummy is contained in each of the dummies3[i]
    small_dummy = dummies1[0] & dummies1[1] & dummies1[2] & dummies1[3]

    intersection = small_dummy & dummies3[0]
    assert intersection == small_dummy

    union = small_dummy | dummies3[0]
    assert union == dummies3[0]

    intersection = small_dummy & ( dummies3[0] | dummies3[1] )
    assert intersection == small_dummy

    intersection = (dummies3[0] | small_dummy) & ( dummies3[0] | dummies3[1] )
    assert intersection == (dummies3[0] | small_dummy)
    #
    # assert intersection == small_dummy
