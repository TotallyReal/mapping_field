import operator

import pytest

from mapping_field.conditions import (
    FalseCondition, IntersectionCondition, NotCondition, TrueCondition, UnionCondition,
    _ListCondition,
)
from mapping_field.ranged_condition import BoolVar
from mapping_field.tests.utils import DummyCondition

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
#       │                Simple construction              │
#       ╰─────────────────────────────────────────────────╯


def test_simple_construction():
    dummy0, dummy1 = DummyCondition(type=0), DummyCondition(type=1)
    UnionCondition([dummy0, dummy1])
    IntersectionCondition([dummy0, dummy1])
    # Remark: The NotCondition(...) is a function calling and not a constructor
    NotCondition(dummy0)


def test_post_generation_independence_not():
    x, y = BoolVar("x"), BoolVar("y")
    func = NotCondition(x)
    assert str(func) == "~(x)"

    # Calling the function
    assigned_func = func({x: y})

    assert assigned_func != func
    assert str(assigned_func) == "~(y)"
    # Some indication that func is frozen
    assert str(func) == "~(x)"


def test_post_generation_independence_and():
    x, y, z = BoolVar("x"), BoolVar("y"), BoolVar("z")
    func = x & y
    assert str(func) == "[y & x]"

    # Calling the function
    assigned_func = func({x: z})

    assert assigned_func != func
    assert str(assigned_func) == "[y & z]"
    # Some indication that func is frozen
    assert str(func) == "[y & x]"


def test_post_generation_independence_or():
    x, y, z = BoolVar("x"), BoolVar("y"), BoolVar("z")
    func = x | y
    assert str(func) == "[y | x]"

    # Calling the function
    assigned_func = func({x: z})

    assert assigned_func != func
    assert str(assigned_func) == "[y | z]"
    # Some indication that func is frozen
    assert str(func) == "[y | x]"


#       ╭─────────────────────────────────────────────────╮
#       │                 List Conditions                 │
#       ╰─────────────────────────────────────────────────╯


@pytest.fixture(params=["Intersection", "Union"])
def dual_case(request):
    op_name = request.param

    dummies1 = [DummyCondition(values={1}, type=i) for i in range(5)]
    dummies2 = [DummyCondition(values={1, 2}, type=i) for i in range(5)]
    if op_name == "Intersection":
        return IntersectionCondition, dummies1, dummies2, operator.and_, operator.or_
    if op_name == "Union":
        return UnionCondition, dummies2, dummies1, operator.or_, operator.and_
    raise NotImplementedError("How did you even get here?!")


# With the following fixtures we have that:
# list_class([weak_dummy, strong_dummy]).simplify2() = bin_op(weak, strong) = strong_dummy


@pytest.fixture
def list_class(dual_case) -> type[_ListCondition]:
    return dual_case[0]


@pytest.fixture
def strong_dummies(dual_case):
    return dual_case[1]


@pytest.fixture
def weak_dummies(dual_case):
    return dual_case[2]


@pytest.fixture
def bin_op(dual_case):
    return dual_case[3]


@pytest.fixture
def rev_bin_op(dual_case):
    return dual_case[4]


def test_intersection_with_delim():
    dummies = [DummyCondition(type=i) for i in range(4)]

    cond1 = IntersectionCondition([dummies[0], dummies[1], dummies[2], dummies[3]])
    cond2 = dummies[0] & dummies[1] & dummies[2] & dummies[3]
    assert cond1 == cond2


def test_union_with_delim():
    dummies = [DummyCondition(type=i) for i in range(4)]

    cond1 = UnionCondition([dummies[0], dummies[1], dummies[2], dummies[3]])
    cond2 = dummies[0] | dummies[1] | dummies[2] | dummies[3]
    assert cond1 == cond2


def test_unpack_lists(list_class: type[_ListCondition]):
    """
    Note that unpacking is done in the constructor without any simplification.
    """
    dummies = [DummyCondition(type=i) for i in range(5)]

    cond1 = list_class([dummies[0], dummies[1]])

    # unpacking one level
    cond2 = list_class([cond1, dummies[2]])
    assert isinstance(cond2, list_class) and len(cond2.conditions) == 3

    cond2 = list_class([dummies[2], cond1])
    assert isinstance(cond2, list_class) and len(cond2.conditions) == 3

    # un packing two levels
    cond3 = list_class([dummies[3], cond2])
    assert isinstance(cond3, list_class) and len(cond3.conditions) == 4


def test_equality_list_permutations(list_class: type[_ListCondition]):
    dummies = [DummyCondition(i) for i in range(5)]

    cond1 = list_class([dummies[0], dummies[1], dummies[2]])
    cond2 = list_class([dummies[2], dummies[0], dummies[1]])
    assert cond1 == cond2


# ------------------ Simplifiers ------------------


def test_simplify_trivial_condition(list_class: type[_ListCondition]):
    dummies = [DummyCondition(type=i) for i in range(5)]

    trivial_condition = list_class.trivials[list_class.type]

    cond1 = list_class([trivial_condition, dummies[0], dummies[1], dummies[2], trivial_condition])
    cond1 = cond1.simplify2()
    cond2 = list_class([dummies[2], dummies[0], dummies[1]])
    assert cond1 == cond2


def test_simplify_final_condition(list_class: type[_ListCondition]):
    dummies = [DummyCondition(type=i) for i in range(5)]

    trivial_condition = list_class.trivials[list_class.type]
    final_condition = list_class.trivials[1 - list_class.type]

    # Controlled by final_condition
    cond1 = list_class([final_condition, dummies[0], dummies[1], dummies[2], trivial_condition])
    cond1 = cond1.simplify2()
    cond2 = final_condition
    assert cond1 == cond2


def test_simplify_repeating_conditions(list_class: type[_ListCondition]):
    dummies = [DummyCondition(type=i) for i in range(5)]

    cond1 = list_class([dummies[0], dummies[1], dummies[1], dummies[0], dummies[2]])
    cond1 = cond1.simplify2()
    cond2 = list_class([dummies[2], dummies[0], dummies[1]])
    assert cond1 == cond2


def test_simplify_unwrapping(list_class: type[_ListCondition]):
    dummies = [DummyCondition(type=i) for i in range(5)]

    cond1 = list_class([dummies[0]])
    cond1 = cond1.simplify2()
    cond2 = dummies[0]
    assert cond1 == cond2


def test_simplify_list_of_lists(list_class: type[_ListCondition]):
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


def test_simplify_containment(list_class: type[_ListCondition]):
    # Test cases like   (A & B & C & D) | (A & B) = (A & B)
    #                   (A | B | C | D) & (A | B) = (A | B)
    rev_op = list_class.op_types[1 - list_class.type]

    dummies = [DummyCondition(values=0, type=i) for i in range(5)]

    conditions = [
        list_class([dummies[0], dummies[1], dummies[2]]),
        list_class([dummies[0], dummies[2]]),
        dummies[2],
    ]

    for i in range(3):
        for j in range(i):
            assert rev_op(conditions[i], conditions[j]) == conditions[i], f"Failed at indices {i=}, {j=}"


def test_proper_containment(
        list_class: type[_ListCondition], weak_dummies: list[DummyCondition], strong_dummies: list[DummyCondition],
        rev_bin_op
):
    """
    Suppose that A | A0 = A0, then we also have (A & B & C ...)| A0 = A0,
    Check both this 1 vs many case, and also add more "noise"
    """

    cond1 = list_class([strong_dummies[0], strong_dummies[1], strong_dummies[2]])
    cond2 = strong_dummies[0]
    cond3 = weak_dummies[0]

    assert rev_bin_op(cond1, cond2) == cond2
    assert rev_bin_op(cond2, cond1) == cond2
    assert rev_bin_op(cond2, cond3) == cond3
    assert rev_bin_op(cond3, cond2) == cond3
    # Transitivity
    assert rev_bin_op(cond1, cond3) == cond3
    assert rev_bin_op(cond3, cond1) == cond3

    # Adding extra "noise"
    noise = list_class([strong_dummies[3], strong_dummies[4]])

    cond1 = list_class([cond1, noise])
    cond2 = list_class([cond2, noise])
    cond3 = list_class([cond3, noise])

    assert rev_bin_op(cond1, cond2) == cond2
    assert rev_bin_op(cond2, cond1) == cond2
    assert rev_bin_op(cond2, cond3) == cond3
    assert rev_bin_op(cond3, cond2) == cond3
    # Transitivity
    assert rev_bin_op(cond1, cond3) == cond3
    assert rev_bin_op(cond3, cond1) == cond3


def test_simplification_list_with_special_union_condition():
    dummies = [DummyCondition(values=0, type=i) for i in range(5)]

    cond1 = dummies[0] & dummies[1] & DummyCondition(values=1, type=2)
    cond2 = dummies[0] & dummies[1] & DummyCondition(values=2, type=2)
    result = dummies[0] & dummies[1] & DummyCondition(values={1, 2}, type=2)

    assert cond1 | cond2 == result


def test_simplification_list_with_special_intersection_condition():
    dummies = [DummyCondition(values=0, type=i) for i in range(5)]

    cond1 = dummies[0] | dummies[1] | DummyCondition(values={1, 2}, type=2)
    cond2 = dummies[0] | dummies[1] | DummyCondition(values={1, 3}, type=2)
    result = dummies[0] | dummies[1] | DummyCondition(values={1}, type=2)

    assert cond1 & cond2 == result


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

    dummies1 = [DummyCondition(values={1}, type=i) for i in range(5)]
    dummies3 = [DummyCondition(values={1, 2, 3}, type=i) for i in range(5)]

    condition1 = dummies3[0] | dummies3[1] | dummies3[2] | dummies3[3]
    condition2 = dummies3[0] | dummies3[1]

    assert condition1 & condition2 == condition2

    # small_dummy is contained in each of the dummies3[i]
    small_dummy = dummies1[0] & dummies1[1] & dummies1[2] & dummies1[3]

    intersection = small_dummy & dummies3[0]
    assert intersection == small_dummy

    union = small_dummy | dummies3[0]
    assert union == dummies3[0]

    intersection = small_dummy & (dummies3[0] | dummies3[1])
    assert intersection == small_dummy

    intersection = (dummies3[0] | small_dummy) & (dummies3[0] | dummies3[1])
    assert intersection == (dummies3[0] | small_dummy)
    #
    # assert intersection == small_dummy

def test_small_big_condition_switch():
    """
    given three conditions A, B_s, B_l with B_s < B_l we have

    (A & B_l) | B_s = (A | B_s) & (B_l | B_s) = (A | B_s) & B_l
    Make sure this doesn't cause a simplification loop.
    """
    cond_big = DummyCondition(type=0, values={0,1})
    cond_small = DummyCondition(type=0, values={0})
    # TODO: Causes problems if we just use x instead of x<<0, since trying to simplify x by setting x==0, causes 0
    #       to appear in the computation instead of FalseCondition. Need to find a way to either 'join' these together
    #       or to know when to treat 0 as a number and when to treat it as a false condition.
    cond1 = BoolVar('x')<<0
    cond = ( ( cond1 & cond_big ) | cond_small )

    cond.simplify2()
