import operator

import pytest

from mapping_field.arithmetics import Add
from mapping_field.conditional_function import ReLU
from mapping_field.conditions import (
    FalseCondition, IntersectionCondition, TrueCondition, UnionCondition,
)
from mapping_field.log_utils.tree_loggers import TreeLogger, blue
from mapping_field.mapping_field import MapElementConstant, Var, simplifier_context
from mapping_field.property_engines import is_integral
from mapping_field.ranged_condition import IntervalRange, RangeCondition, in_range, XX
from mapping_field.tests.utils import DummyMap
from itertools import product

simplify_logger = TreeLogger(__name__)


#       ╭─────────────────────────────────────────────────╮
#       │                Interval tests                   │
#       ╰─────────────────────────────────────────────────╯

def test_interval_generation():
    points = [float("-inf"), -5, 5, float("inf")]
    booleans = [True, False]
    for low, high, contains_low, contains_high in product(points, points, booleans, booleans):
        interval = IntervalRange(low, high, contains_low, contains_high)
    for low, high in product(points, points):
        interval = IntervalRange(low, high)
        interval = IntervalRange[low, high]


def test_unique_interval_generation():
    empty_interval = IntervalRange(1, 0, False, False)
    assert empty_interval.is_empty
    empty_interval = IntervalRange.empty()
    assert empty_interval.is_empty

    all_interval = IntervalRange(float("-inf"), float("inf"), False, False)
    assert all_interval.is_all
    all_interval = IntervalRange.all()
    assert all_interval.is_all

    point_interval = IntervalRange(5, 5, True, True)
    assert point_interval.is_point == 5
    point_interval = IntervalRange.of_point(5)
    assert point_interval.is_point == 5


def test_interval_equality():
    # Half open intervals
    interval1 = IntervalRange(5, 8, True, False)
    interval2 = IntervalRange(5, 8, True, False)
    assert interval1 == interval2

    interval2 = IntervalRange(5, 8)
    assert interval1 == interval2

    # Close intervals
    interval1 = IntervalRange(5, 8, True, True)
    interval2 = IntervalRange(5, 8, True, True)
    assert interval1 == interval2

    interval2 = IntervalRange[5, 8]
    assert interval1 == interval2
    assert interval1.is_closed()

    assert IntervalRange(5, 8, False, False).is_open()

    # Empty intervals
    assert IntervalRange.empty() == IntervalRange(1, -1)

    # Full intervals
    assert IntervalRange.all() == IntervalRange(float("-inf"), float("inf"))

    # Point interval
    assert IntervalRange.of_point(5) == IntervalRange[5, 5]


def test_interval_xx_construction():
    assert (XX < 5)  == IntervalRange(float("-inf"), 5, False, False)
    assert (XX <= 5) == IntervalRange(float("-inf"), 5, False, True)
    assert (5 < XX)  == IntervalRange(5, float("inf"), False, False)
    assert (5 <= XX) == IntervalRange(5, float("inf"), True, False)

    assert (5 <= XX) & (XX < 10) == IntervalRange(5, 10, True, False)
    assert (XX << 5) == IntervalRange.of_point(5)

#       ╭─────────────────────────────────────────────────╮
#       │               Ranged Condition                  │
#       ╰─────────────────────────────────────────────────╯

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


def test_trivial_ranges():
    dummy = DummyMap(0)

    cond = RangeCondition(dummy, IntervalRange.empty()).simplify()
    assert cond is FalseCondition

    cond = RangeCondition(dummy, IntervalRange.all()).simplify()
    assert cond is TrueCondition


def test_ranged_comparison_from_interval_comparison():
    dummy = DummyMap(0)

    for op in (operator.lt, operator.le, operator.gt, operator.ge, operator.lshift):
        assert op(dummy, 5) == RangeCondition(dummy, op(XX, 5)), \
            f"{op} does not transform from RangedCondition to Interval"

    # Unfortunately, python is terrible, and I can't use 2-sided comparisons like:
    #   cond = (10 <= dummy < 20)
    cond = (10 <= dummy) & (dummy < 20)
    assert cond == RangeCondition(dummy, (10 <= XX) & (XX < 20))

    cond = dummy << 10
    assert cond == RangeCondition(dummy, XX << 10) == (dummy.where() == 10)


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


def test_range_boolean_derived_from_intervals(monkeypatch):
    dummy = DummyMap(0)
    cond1 = RangeCondition(dummy, (0, 10))
    cond2 = RangeCondition(dummy, (5, 17))

    range_union = IntervalRange[0,1]
    monkeypatch.setattr(IntervalRange, "union", lambda self, other: range_union)
    cond = cond1 | cond2
    assert isinstance(cond, RangeCondition) and (cond.function is dummy) and (cond.range is range_union)

    range_intersection = IntervalRange[1,2]
    monkeypatch.setattr(IntervalRange, "intersection", lambda self, other: range_intersection)
    cond = cond1 & cond2
    assert isinstance(cond, RangeCondition) and (cond.function is dummy) and (cond.range is range_intersection)


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
    assert cond12._simplify() is None

    # cond2 = SingleAssignmentCondition(dummy_var, 10)
    # cond12 = RangeCondition(dummy_var, (0,11))
    # assert cond1 | cond2 == cond12


def test_simplify_all_or_nothing_range():
    dummy_map = DummyMap(0)

    cond = RangeCondition(dummy_map, (10, 10))
    assert cond is not FalseCondition
    assert cond.simplify() is FalseCondition

    cond = RangeCondition(dummy_map, (20, 10))
    assert cond is not FalseCondition
    assert cond.simplify() is FalseCondition

    cond = RangeCondition(dummy_map, (float("-inf"), float("+inf")))
    assert cond is not TrueCondition
    assert cond.simplify() is TrueCondition


def test_simplify_evaluated():
    c = Add(MapElementConstant(3), MapElementConstant(2), simplify=False)
    assert str(c) != str(MapElementConstant(5))
    # c is not by itself a constant, but can be evaluated into a constant

    assert (c < 10).simplify() == TrueCondition
    assert (1 <= c).simplify() == TrueCondition
    assert (c << 5).simplify() == TrueCondition
    assert (RangeCondition(c, (1, 10))).simplify() == TrueCondition

    assert (c > 10).simplify() == FalseCondition
    assert (1 >= c).simplify() == FalseCondition
    assert (c << 8).simplify() == FalseCondition
    assert (RangeCondition(c, (10, 100))).simplify() == FalseCondition


def test_simplify_linear_ranged_condition():
    dummy = DummyMap(0)

    def inner_test(a, b, low, high) -> None:
        cond1 = RangeCondition(dummy, (low, high))
        low, high = a * low + b, a * high + b
        if a < 0:
            cond2 = RangeCondition(a * dummy + b, IntervalRange(high, low, False, True))
        else:
            cond2 = RangeCondition(a * dummy + b, (low, high))

        cond2 = cond2.simplify()
        assert cond1 == cond2

    inner_test(a=2, b=3, low=1, high=6)
    inner_test(a=-2, b=3, low=-5, high=0)

    # test zero coefficient - False
    condition = RangeCondition(0 * dummy + 3, (5, 15))
    condition = condition.simplify()
    assert condition == FalseCondition

    # test zero coefficient - False
    condition = RangeCondition(0 * dummy + 7, (5, 15))
    condition = condition.simplify()
    assert condition == TrueCondition


def test_ranged_condition_as_input():
    x, y, z = Var("x"), Var("y"), Var("z")
    a, b = Var("a"), Var("b")

    func = (x + y) + z
    condition = (x << 1) & (y << 2) & (z << 3) & (a << 4) & (b < 10)
    assigned = func(condition=condition, simplify=False)
    assert str(assigned) == "(1 + 2 + 3)"
    assert assigned.simplify() == 6


def test_sum_of_conditions():
    n = 2
    simplify_logger.tree.max_log_count = -1
    x = [Var(f'x_{i}') for i in range(n)]
    # simplify_logger.tree.set_active(False)
    elem = sum([x[i] << 2 * i for i in range(n)], 1 - n)

    cond1 = (0<elem).simplify()
    cond2 = IntersectionCondition([x[i] << 2 * i for i in range(n)])
    assert cond1 == cond2

    elem = ReLU(elem)
    # simplify_logger.tree.set_active(True)
    cond1 = elem.simplify()
    assert cond1 == cond2


#       ╭─────────────────────────────────────────────────╮
#       │               IsIntegral promise                │
#       ╰─────────────────────────────────────────────────╯


def test_equality_as_integral():
    dummy = DummyMap(0)

    cond1 = (4 < dummy) & (dummy <= 10.2)
    cond2 = (4.8 <= dummy) & (dummy <= 10.4)
    assert cond1 != cond2

    dummy = DummyMap(output_properties={is_integral: True})

    cond1 = (4 < dummy) & (dummy <= 10.2)
    cond2 = (4.8 <= dummy) & (dummy <= 10.4)
    assert cond1 == cond2

    cond1 = (4 < dummy) & (dummy <= 5.2)
    cond2 = dummy << 5
    assert cond1 == cond2


def test_union_for_integral_functions():
    dummy = DummyMap(0)

    cond1 = (5.5 <= dummy) & (dummy <= 10.2)
    cond2 = (10.8 <= dummy) & (dummy <= 17.4)
    result = UnionCondition([cond1, cond2])._simplify()
    assert result is None

    dummy = DummyMap(output_properties={is_integral: True})

    cond1 = (5.5 <= dummy) & (dummy <= 10.2)
    cond2 = (10.8 <= dummy) & (dummy <= 17.4)
    result = UnionCondition([cond1, cond2])._simplify()
    assert result == (6 <= dummy) & (dummy <= 17)


def test_union_of_integral_points():
    dummy = DummyMap(output_properties={is_integral: True})

    conditions = [(dummy << i) for i in range(3, 9)]
    union = UnionCondition(conditions).simplify()

    result = (3 <= dummy) & (dummy <= 8)
    assert union == result

#       ╭─────────────────────────────────────────────────╮
#       │               Ranged Condition                  │
#       ╰─────────────────────────────────────────────────╯


def test_integral_product():
    dummy = DummyMap(output_properties={is_integral: True})
    dummy2 = 2 * dummy  # Only even integers

    cond1 = (6.5 <= dummy2).simplify()
    cond2 = (8 <= dummy2).simplify()
    cond3 = (10 <= dummy2).simplify()
    assert cond1 == cond2
    assert cond1 != cond3

    dummy = DummyMap(output_properties={is_integral: True, in_range: 5 <= XX})
    dummy2 = 2 * dummy

    # Now dummy is an integer >=5, so that dummy2 is an even integer >= 10
    cond1 = (6.5 <= dummy2).simplify()
    cond2 = (10 <= dummy2).simplify()
    assert cond1 == cond2


def test_simplify_on_ranged_promised_functions():
    dummy = DummyMap(0)

    # Simple condition, can't be simplified.
    condition = (5 <= dummy)
    condition = condition._simplify()
    assert condition is None

    # Add to dummy a nonnegative output assumption
    dummy = DummyMap(output_properties={in_range: 0 <= XX})

    condition = (-5 <= dummy)
    condition = condition.simplify()
    assert condition is TrueCondition

    condition = dummy < -5
    condition = condition.simplify()
    assert condition is FalseCondition

    # new condition is contained inside the assumption, so no change
    condition = 5 <= dummy
    condition = condition._simplify()
    assert condition is None

    # intersection of new condition and assumption
    condition = dummy < 5
    condition = condition.simplify()
    result = RangeCondition(dummy, (0, 5))
    assert condition is not result


def test_sum_of_two_conditions():
    dummy = [DummyMap(value=i, output_properties={in_range: IntervalRange[0, 1]}) for i in (0, 1)]

    cond1 = (dummy[0] + dummy[1]) << 2
    cond1 = cond1.simplify()
    cond2 = (dummy[0] << 1) & (dummy[1] << 1)

    assert cond1 == cond2


def test_add_ranged_equality():
    dummy1 = DummyMap(1, output_properties={in_range: IntervalRange[0, 1]})
    dummy2 = DummyMap(2, output_properties={in_range: IntervalRange[0, 1]})

    cond1 = ((dummy1 + dummy2) << 2).simplify()
    cond2 = (dummy1 << 1) & (dummy2 << 1)
    assert cond1 == cond2


def test_sub_ranged_equality():
    dummy1 = DummyMap(1, output_properties={in_range: IntervalRange[0, 1]})
    dummy2 = DummyMap(2, output_properties={in_range: IntervalRange[0, 1]})

    cond1 = ((dummy1 - dummy2) << 1).simplify()
    cond2 = (dummy1 << 1) & (dummy2 << 0)
    assert cond1 == cond2


def test_addition_arithmetic_and_ranged():
    dummy1 = DummyMap(1, output_properties={in_range: IntervalRange[0, 1], is_integral: True})
    dummy2 = DummyMap(2, output_properties={in_range: IntervalRange[0, 1], is_integral: True})

    cond = RangeCondition(dummy1 - dummy2, IntervalRange(0, 1, False, True))
    cond = cond.simplify()
    assert cond == ((dummy1 << 1) & (dummy2 << 0))


def test_intersection_range_property_with_condition():
    dummy = DummyMap(output_properties={in_range: IntervalRange[0, 10]})

    cond = dummy < 100
    assert cond.simplify() is TrueCondition

    cond = dummy > 100
    assert cond.simplify() is FalseCondition

    cond = (3 <= dummy) & (dummy <= 5)
    cond.simplify()
    assert isinstance(cond, RangeCondition) and (cond.function is dummy) and (cond.range == IntervalRange[3, 5])

    cond = (3 <= dummy) & (dummy <= 15)
    result = (3 <= dummy) & (dummy <= 10)
    assert cond.simplify() == result


def test_in_range_promise():
    dummy = DummyMap()

    cond1 = (dummy << 0) | (dummy << 1)
    cond2 = (0 <= dummy) & (dummy <= 1)
    cond3 = RangeCondition(dummy, IntervalRange[0, 1])

    assert cond1.simplify() != TrueCondition
    assert cond2.simplify() != TrueCondition
    assert cond3.simplify() != TrueCondition

    # Now make it only take values in 0 or 1 (namely BoolVar)
    simplify_logger.log(blue("Adding 'condition' property"))
    dummy = DummyMap(output_properties={in_range: IntervalRange[0, 1], is_integral: True})

    cond1 = (dummy << 0) | (dummy << 1)
    cond2 = (0 <= dummy) & (dummy <= 1)
    cond3 = RangeCondition(dummy, IntervalRange[0, 1])

    assert cond1.simplify() is TrueCondition
    assert cond2.simplify() is TrueCondition
    assert cond3.simplify() is TrueCondition


#       ╭─────────────────────────────────────────────────╮
#       │             in_range engine rules               │
#       ╰─────────────────────────────────────────────────╯


class TestInRange:

    def test_constant_in_range(self):
        c = MapElementConstant(5)

        assert in_range.compute(c, simplifier_context) == IntervalRange.of_point(5)


    def test_addition_in_range(self):
        dummy1 = DummyMap(1, output_properties={in_range: IntervalRange[1, 4]})
        dummy2 = DummyMap(2, output_properties={in_range: IntervalRange[3, 5]})
        dummy3 = DummyMap(2, output_properties={in_range: IntervalRange[-7, 7]})

        result = dummy1 + dummy2 + dummy3
        f_range = in_range.compute(result, simplifier_context)
        assert f_range is not None
        assert f_range == IntervalRange[-3, 16]


    def test_scalar_mult_in_range(self):
        interval = IntervalRange[1,4]
        dummy = DummyMap(1, output_properties={in_range: interval})

        f_range = in_range.compute(3 * dummy, simplifier_context)
        assert f_range is not None
        assert f_range == 3 * interval

        f_range = in_range.compute(-3 * dummy, simplifier_context)
        assert f_range is not None
        assert f_range == -3 * interval

        f_range = in_range.compute(0 * dummy, simplifier_context)
        assert f_range is not None
        assert f_range == 0 * interval