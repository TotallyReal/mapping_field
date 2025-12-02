from mapping_field.arithmetics import Add
from mapping_field.conditional_function import ReLU
from mapping_field.conditions import (
    FalseCondition, IntersectionCondition, TrueCondition, UnionCondition,
)
from mapping_field.log_utils.tree_loggers import TreeLogger
from mapping_field.mapping_field import MapElementConstant, Var, simplifier_context
from mapping_field.promises import IsIntegral
from mapping_field.property_engines import is_integral
from mapping_field.ranged_condition import InRange, IntervalRange, RangeCondition, in_range
from mapping_field.tests.utils import DummyMap

simplify_logger = TreeLogger(__name__)

def test_in_range_promise():
    dummy = DummyMap()

    cond1 = (dummy << 0) | (dummy << 1)
    cond2 = (0 <= dummy) & (dummy <= 1)
    cond3 = RangeCondition(dummy, IntervalRange[0, 1])

    assert cond1.simplify2() != TrueCondition
    assert cond2.simplify2() != TrueCondition
    assert cond3.simplify2() != TrueCondition

    # Now make it only take values in 0 or 1 (namely BoolVar)
    dummy.promises.add_promise(InRange(IntervalRange[0, 1]))
    simplifier_context.set_property(dummy, is_integral, True)

    cond1 = (dummy << 0) | (dummy << 1)
    cond2 = (0 <= dummy) & (dummy <= 1)
    cond3 = RangeCondition(dummy, IntervalRange[0, 1])

    assert cond1.simplify2() is TrueCondition
    assert cond2.simplify2() is TrueCondition
    assert cond3.simplify2() is TrueCondition


# TODO: Most of the logic moved to the IntervalRange class, so the test should move there as well.


def test_simple_construction():
    dummy = DummyMap(0)
    RangeCondition(dummy, IntervalRange(0, 1))


def test_post_generation_independence():
    x = Var("x")
    f_range = IntervalRange(0, 1)
    func = RangeCondition(x, f_range)
    assert str(func) == "0<=x<1"

    # TODO: Both range and x should be frozen. Keeping this reminder here until I actually make them such.

    # Calling the function
    assigned_func = func({x: DummyMap(0)})

    assert assigned_func != func
    assert str(assigned_func) == "0<=DummyMap(0)<1"
    # Some indication that func is frozen
    assert str(func) == "0<=x<1"


def test_comparison_operators():
    dummy = DummyMap(0)

    cond = dummy < 10
    assert cond == RangeCondition(dummy, IntervalRange(float("-inf"), 10, False, False))

    cond = dummy <= 10
    assert cond == RangeCondition(dummy, IntervalRange(float("-inf"), 10, False, True))

    cond = 10 <= dummy
    assert cond == RangeCondition(dummy, IntervalRange(10, float("inf"), True, False))

    cond = 10 < dummy
    assert cond == RangeCondition(dummy, IntervalRange(10, float("inf"), False, False))

    # Unfortunately, python is terrible, and I can't use 2-sided comparisons like:
    #   cond = (10 <= dummy < 20)


def test_lshift_operator():
    dummy = DummyMap(0)

    cond1 = dummy << 5
    cond2 = RangeCondition(dummy, IntervalRange.of_point(5))
    cond3 = dummy.where() == 5

    assert cond1 == cond2 == cond3


def test_invert_range():
    dummy = DummyMap(0)

    condition1 = ~(dummy < 3)
    condition2 = dummy >= 3
    assert condition1 == condition2

    condition1 = ~(dummy <= 3)
    condition2 = dummy > 3
    assert condition1 == condition2

    condition1 = ~RangeCondition(dummy, (1, 3))
    condition2 = (dummy < 1) | (3 <= dummy)
    assert condition1 == condition2


def test_range_condition_intersection():
    dummy_map = DummyMap(0)

    # containment
    cond1 = RangeCondition(dummy_map, (0, 10))
    cond2 = RangeCondition(dummy_map, (5, 7))
    assert cond1 & cond2 == cond2

    # closed ranges
    cond1 = RangeCondition(dummy_map, (0, 10))
    cond2 = RangeCondition(dummy_map, (5, 15))
    cond12 = RangeCondition(dummy_map, (5, 10))
    assert cond1 & cond2 == cond12

    # half open ranges
    cond1 = 5 <= dummy_map
    cond2 = dummy_map < 10
    cond12 = RangeCondition(dummy_map, (5, 10))
    assert cond1 & cond2 == cond12

    # disjoint ranges
    cond1 = RangeCondition(dummy_map, (0, 10))
    cond2 = RangeCondition(dummy_map, (15, 25))
    assert cond1 & cond2 == FalseCondition


def test_range_condition_union():
    dummy_map = DummyMap(0)

    # containment
    cond1 = RangeCondition(dummy_map, (0, 10))
    cond2 = RangeCondition(dummy_map, (5, 7))
    assert cond1 | cond2 == cond1

    # Union with intersection
    cond1 = RangeCondition(dummy_map, (0, 10))
    cond2 = RangeCondition(dummy_map, (5, 15))
    cond12 = RangeCondition(dummy_map, (0, 15))
    assert cond1 | cond2 == cond12

    # Consecutive ranges with no intersection
    cond1 = RangeCondition(dummy_map, (0, 8))
    cond2 = RangeCondition(dummy_map, (8, 15))
    cond12 = RangeCondition(dummy_map, (0, 15))
    assert cond1 | cond2 == cond12

    # Purely disjoint ranges
    cond1 = RangeCondition(dummy_map, (0, 5))
    cond2 = RangeCondition(dummy_map, (10, 15))
    cond12 = UnionCondition([cond1, cond2])
    assert cond12._simplify2() is None

    # cond2 = SingleAssignmentCondition(dummy_var, 10)
    # cond12 = RangeCondition(dummy_var, (0,11))
    # assert cond1 | cond2 == cond12


def test_simplify_all_or_nothing_range():
    dummy_map = DummyMap(0)

    cond = RangeCondition(dummy_map, (10, 10))
    assert cond is not FalseCondition
    assert cond.simplify2() is FalseCondition

    cond = RangeCondition(dummy_map, (20, 10))
    assert cond is not FalseCondition
    assert cond.simplify2() is FalseCondition

    cond = RangeCondition(dummy_map, (float("-inf"), float("+inf")))
    assert cond is not TrueCondition
    assert cond.simplify2() is TrueCondition


def test_simplify_evaluated():
    c = Add(MapElementConstant(3), MapElementConstant(2), simplify=False)
    assert str(c) != str(MapElementConstant(5))
    # c is not by itself a constant, but can be evaluated into a constant

    assert (c < 10).simplify2() == TrueCondition
    assert (1 <= c).simplify2() == TrueCondition
    assert (c << 5).simplify2() == TrueCondition
    assert (RangeCondition(c, (1, 10))).simplify2() == TrueCondition

    assert (c > 10).simplify2() == FalseCondition
    assert (1 >= c).simplify2() == FalseCondition
    assert (c << 8).simplify2() == FalseCondition
    assert (RangeCondition(c, (10, 100))).simplify2() == FalseCondition


def test_simplify_linear_ranged_condition():
    dummy = DummyMap(0)

    def inner_test(a, b, low, high) -> None:
        cond1 = RangeCondition(dummy, (low, high))
        low, high = a * low + b, a * high + b
        if a < 0:
            cond2 = RangeCondition(a * dummy + b, IntervalRange(high, low, False, True))
        else:
            cond2 = RangeCondition(a * dummy + b, (low, high))

        cond2 = cond2.simplify2()
        assert cond1 == cond2

    inner_test(a=2, b=3, low=1, high=6)
    inner_test(a=-2, b=3, low=-5, high=0)

    # test zero coefficient - False
    condition = RangeCondition(0 * dummy + 3, (5, 15))
    condition = condition.simplify2()
    assert condition == FalseCondition

    # test zero coefficient - False
    condition = RangeCondition(0 * dummy + 7, (5, 15))
    condition = condition.simplify2()
    assert condition == TrueCondition


def test_simplify_on_ranged_promised_functions():
    dummy = DummyMap(0)

    # Simple condition, can't be simplified.
    condition = 5 <= dummy
    condition = condition._simplify2()
    assert condition is None

    # Add to dummy a nonnegative output assumption
    dummy.promises.add_promise(InRange((0, float("inf"))))

    condition = -5 <= dummy
    condition = condition.simplify2()
    assert condition is TrueCondition

    condition = dummy < -5
    condition = condition.simplify2()
    assert condition is FalseCondition

    # new condition is contained inside the assumption, so no change
    condition = 5 <= dummy
    condition = condition._simplify2()
    assert condition is None

    # intersection of new condition and assumption
    condition = dummy < 5
    condition = condition.simplify2()
    result = RangeCondition(dummy, (0, 5))
    assert condition is not result


#       ╭─────────────────────────────────────────────────╮
#       │               IsIntegral promise                │
#       ╰─────────────────────────────────────────────────╯


def test_equality_as_integral():
    dummy = DummyMap(0)

    cond1 = (4 < dummy) & (dummy <= 10.2)
    cond2 = (4.8 <= dummy) & (dummy <= 10.4)
    assert cond1 != cond2

    simplifier_context.set_property(dummy, is_integral, True)

    cond1 = (4 < dummy) & (dummy <= 10.2)
    cond2 = (4.8 <= dummy) & (dummy <= 10.4)
    assert cond1 == cond2

    cond1 = (4 < dummy) & (dummy <= 5.2)
    cond2 = dummy << 5
    assert cond1 == cond2


def test_integral_product():
    dummy = DummyMap(output_properties={is_integral: True})
    dummy2 = 2 * dummy  # Only even integers

    cond1 = (6.5 <= dummy2).simplify2()
    cond2 = (8 <= dummy2).simplify2()
    cond3 = (10 <= dummy2).simplify2()
    assert cond1 == cond2
    assert cond1 != cond3

    dummy.promises.add_promise(InRange(IntervalRange(5, float("inf"), True, False)))
    # Now dummy is an integer >=5, so that dummy2 is an even integer >= 10
    cond1 = (6.5 <= dummy2).simplify2()
    cond2 = (10 <= dummy2).simplify2()
    assert cond1 == cond2


def test_union_for_integral_functions():
    dummy = DummyMap(0)

    cond1 = (5.5 <= dummy) & (dummy <= 10.2)
    cond2 = (10.8 <= dummy) & (dummy <= 17.4)
    result = UnionCondition([cond1, cond2])._simplify2()
    assert result is None

    dummy = DummyMap(output_properties={is_integral: True})

    cond1 = (5.5 <= dummy) & (dummy <= 10.2)
    cond2 = (10.8 <= dummy) & (dummy <= 17.4)
    result = UnionCondition([cond1, cond2])._simplify2()
    assert result == (6 <= dummy) & (dummy <= 17)


def test_union_of_integral_points():
    dummy = DummyMap(output_properties={is_integral: True})

    conditions = [(dummy << i) for i in range(3, 9)]
    union = UnionCondition(conditions).simplify2()

    result = (3 <= dummy) & (dummy <= 8)
    assert union == result


def test_ranged_condition_as_input():
    x, y, z = Var("x"), Var("y"), Var("z")
    a, b = Var("a"), Var("b")

    func = (x + y) + z
    condition = (x << 1) & (y << 2) & (z << 3) & (a << 4) & (b < 10)
    assigned = func(condition=condition, simplify=False)
    assert str(assigned) == "(1 + 2 + 3)"
    assert assigned.simplify2() == 6


def test_range_of_constant():
    c = MapElementConstant(5)

    assert in_range.compute(c, simplifier_context) == IntervalRange.of_point(5)

def test_sum_of_two_conditions():
    dummy0, dummy1 = DummyMap(0), DummyMap(1)

    dummy0.promises.add_promise(InRange(IntervalRange[0,1]))
    dummy1.promises.add_promise(InRange(IntervalRange[0,1]))

    cond1 = (dummy0 + dummy1) << 2
    cond1 = cond1.simplify2()
    cond2 = (dummy0 << 1) & (dummy1 << 1)

    assert cond1 == cond2

def test_sum_of_conditions():
    n = 2
    simplify_logger.tree.max_log_count = -1
    x = [Var(f'x_{i}') for i in range(n)]
    # simplify_logger.tree.set_active(False)
    elem = sum([x[i]<<2*i for i in range(n)], 1-n)
    elem = ReLU(elem)
    # simplify_logger.tree.set_active(True)
    cond1 = elem.simplify2()
    cond2 = IntersectionCondition([x[i]<<2*i for i in range(n)])
    assert cond1 == cond2

def test_add_ranged_functions():
    dummy1, dummy2 = DummyMap(1), DummyMap(2)

    dummy1.promises.add_promise(InRange(IntervalRange[1, 4]))
    dummy2.promises.add_promise(InRange(IntervalRange[3, 5]))
    result = dummy1 + dummy2
    f_range = in_range.compute(result, simplifier_context)
    assert f_range is not None
    assert f_range == IntervalRange[4, 9]


def test_sub_ranged_functions():
    dummy1, dummy2 = DummyMap(1), DummyMap(2)

    dummy1.promises.add_promise(InRange(IntervalRange[1, 4]))
    dummy2.promises.add_promise(InRange(IntervalRange[3, 5]))
    result = dummy1 - dummy2
    f_range = in_range.compute(result, simplifier_context)
    assert f_range is not None
    assert f_range == IntervalRange[-4, 1]


def test_add_ranged_equality():
    dummy1, dummy2 = DummyMap(1), DummyMap(2)

    dummy1.promises.add_promise(InRange(IntervalRange[0, 1]))
    dummy2.promises.add_promise(InRange(IntervalRange[0, 1]))

    cond1 = ((dummy1 + dummy2) << 2).simplify2()
    cond2 = (dummy1 << 1) & (dummy2 << 1)
    assert cond1 == cond2


def test_sub_ranged_equality():
    dummy1, dummy2 = DummyMap(1), DummyMap(2)

    dummy1.promises.add_promise(InRange(IntervalRange[0, 1]))
    dummy2.promises.add_promise(InRange(IntervalRange[0, 1]))

    cond1 = ((dummy1 - dummy2) << 1).simplify2()
    cond2 = (dummy1 << 1) & (dummy2 << 0)
    assert cond1 == cond2


def test_addition_arithmetic_and_ranged():
    dummy1 = DummyMap(1, output_properties={is_integral: True})
    dummy2 = DummyMap(2, output_properties={is_integral: True})


    dummy1.promises.add_promise(InRange(IntervalRange[0, 1]))
    dummy2.promises.add_promise(InRange(IntervalRange[0, 1]))

    cond = RangeCondition(dummy1 - dummy2, IntervalRange(0, 1, False, True))
    cond = cond.simplify2()
    assert cond == ((dummy1 << 1) & (dummy2 << 0))
