import logging

from mapping_field.binary_expansion import BinaryExpansion
from mapping_field.conditional_function import ConditionalFunction, ReLU
from mapping_field.conditions import FalseCondition, TrueCondition
from mapping_field.linear import Linear
from mapping_field.mapping_field import MapElementConstant, Var, simplifier_context
from mapping_field.promises import IsIntegral
from mapping_field.property_engines import is_integral
from mapping_field.ranged_condition import BoolVar, InRange, IntervalRange, RangeCondition, in_range
from mapping_field.tests.utils import DummyCondition, DummyConditionOn, DummyMap

logger = logging.getLogger(__name__)


def test_simple_construction():
    dummy = DummyMap(0)
    cond_func = ConditionalFunction([(dummy < 0, dummy)])
    assert str(cond_func) == "[ DummyMap(0)<0 -> DummyMap(0) ]"


def test_post_generation_independence():
    dummy0 = DummyMap(0)
    x = Var("x")
    regions = [
        (dummy0 < 0, x),
        (x < 0, dummy0),
    ]
    func = ConditionalFunction(regions)
    assert str(func) == "[ DummyMap(0)<0 -> x  ;  x<0 -> DummyMap(0) ]"

    # Changing the region list should not change the function
    dummy1 = DummyMap(1)
    regions[0] = (dummy0 < 0, dummy1)
    regions[1] = (dummy1 < 0, dummy0)
    region_changed_func = ConditionalFunction(regions)

    assert str(func) == "[ DummyMap(0)<0 -> x  ;  x<0 -> DummyMap(0) ]"
    assert str(region_changed_func) == "[ DummyMap(0)<0 -> DummyMap(1)  ;  DummyMap(1)<0 -> DummyMap(0) ]"

    # Calling the function
    assigned_func = func({x: dummy1})

    assert str(assigned_func) == "[ DummyMap(0)<0 -> DummyMap(1)  ;  DummyMap(1)<0 -> DummyMap(0) ]"
    # Some indication that func is frozen
    assert str(func) == "[ DummyMap(0)<0 -> x  ;  x<0 -> DummyMap(0) ]"


def test_evaluate():
    dummy_conditions = [DummyCondition(i) for i in range(3)]

    func = ConditionalFunction([
        (dummy_conditions[0], MapElementConstant(5)),
        (dummy_conditions[1], MapElementConstant(5)),
        (dummy_conditions[2], MapElementConstant(5)),
    ])
    assert func.evaluate() == 5

    func = ConditionalFunction([
        (dummy_conditions[0], MapElementConstant(3)),
        (dummy_conditions[1], MapElementConstant(5)),
        (dummy_conditions[2], MapElementConstant(5)),
    ])
    assert func.evaluate() is None

    func = ConditionalFunction([
        (dummy_conditions[0], MapElementConstant(3)),
        (dummy_conditions[1], DummyMap(0)),
    ])
    assert func.evaluate() is None


def test_equality_to_standard_function():
    dummy_conditions = [DummyConditionOn(set_size=3, values=i) for i in range(3)]
    dummy = DummyMap(0)

    cond_func = ConditionalFunction([
        (dummy_conditions[0], dummy),
        (dummy_conditions[1], dummy),
        (dummy_conditions[2], dummy),
    ])

    assert cond_func.simplify2() is dummy

    cond_func = ConditionalFunction([
        (dummy_conditions[0], dummy),
        (dummy_conditions[1], dummy),
        (dummy_conditions[2], 0),
    ])

    assert cond_func.simplify2() != dummy


def test_equality_to_conditional_function():

    dummies = [DummyConditionOn(set_size=3, values={i}) for i in range(3)]  # Two distinct dummies do not intersect
    cc = [MapElementConstant(i) for i in range(3)]

    cond_func1 = ConditionalFunction([
        (dummies[0], cc[0]),
        (dummies[1], cc[1]),
        (dummies[2], cc[2]),
    ])
    cond_func2 = ConditionalFunction([
        (dummies[0], cc[0]),
        (dummies[1], cc[1]),
        (dummies[2], cc[2]),
    ])

    assert cond_func1 == cond_func2


def test_combining_regions():
    x, y = BoolVar("x"), BoolVar("y")

    func = ConditionalFunction([
            ((x << 0) & (y << 0), MapElementConstant(0)),
            ((x << 1) & (y << 0), MapElementConstant(1)),
            ((y << 1), x),
    ])

    func = func.simplify2()
    assert func is x


def test_equality_region_wise():
    x = Var("x")

    cond_func = ConditionalFunction([
        (x << 0, MapElementConstant(0)),
        (x << 2, MapElementConstant(2)),
        ((7 <= x) & (x < 17), x),
    ])

    assert cond_func == x


def test_addition():
    dummies = [DummyConditionOn(set_size=4, values={i}) for i in range(4)]

    cond_func1 = ConditionalFunction([
        (dummies[0] | dummies[1], MapElementConstant(0)),
        (dummies[2], MapElementConstant(10)),
        # TODO: test fails when removing this last region, since the simplification process changes it from a
        #       conditional function into linear combination of conditions.
        (dummies[3], MapElementConstant(-10)),
    ])

    cond_func2 = ConditionalFunction([
        (dummies[0], MapElementConstant(100)),
        (dummies[1] | dummies[2], MapElementConstant(200)),
        (dummies[3], MapElementConstant(-10)),
    ])

    cond_add = cond_func1 + cond_func2

    result = ConditionalFunction([
        (dummies[0], MapElementConstant(100)),
        (dummies[1], MapElementConstant(200)),
        (dummies[2], MapElementConstant(210)),
        (dummies[3], MapElementConstant(-20)),
    ])

    result = result.simplify2()

    assert result == cond_add, f"could not match:\n{result}\n{cond_add}"


def test_addition_with_ranges():
    dummy_map = DummyMap(0, output_properties={in_range: IntervalRange(0,40,True,False)})

    def ranged(low, high):
        return RangeCondition(dummy_map, IntervalRange(low,high,True,False))

    cond_func1 = ConditionalFunction([
        (ranged(0,10), MapElementConstant(0)),
        (ranged(10,30), MapElementConstant(10)),
        # TODO: test fails when removing this last region, since the simplification process changes it from a
        #       conditional function into linear combination of conditions.
        (ranged(30,40), MapElementConstant(-10))
    ])

    cond_func2 = ConditionalFunction([
        (ranged(0,20), MapElementConstant(100)),
        (ranged(20,30), MapElementConstant(200)),
        (ranged(30,40), MapElementConstant(-10))
    ])

    cond_add = cond_func1 + cond_func2

    result = ConditionalFunction([
        (ranged(10,20), MapElementConstant(110)),
        (ranged(0,10), MapElementConstant(100)),
        (ranged(20,30), MapElementConstant(210)),
        (ranged(30,40), MapElementConstant(-20))
    ])

    assert result == cond_add


def test_simplification():

    # combine regions with the same function
    dummy_cond = [DummyConditionOn(set_size = 3, values={i}) for i in range(3)]
    dummy_func = [DummyMap(i) for i in range(3)]

    cond_func = ConditionalFunction([
        (dummy_cond[0], dummy_func[0]),
        (dummy_cond[1], dummy_func[1]),
        (dummy_cond[2], dummy_func[0]),
    ])
    cond_func = cond_func.simplify2()

    simplified_version = ConditionalFunction([
        (dummy_cond[0] | dummy_cond[2], dummy_func[0]),
        (dummy_cond[1], dummy_func[1]),
    ])

    assert cond_func == simplified_version

    # Combine regions with assignments
    x = Var("x", output_properties={in_range: IntervalRange[0,10]})
    xx = Linear.of(x)

    cond_func = ConditionalFunction([
        ( (0<=x) & (x<10), xx + 3),
        (x << 10, MapElementConstant(13)),
    ])

    cond_func = cond_func.simplify2()

    assert cond_func == xx + 3


def test_linear_ranged_condition_subtraction():
    vv = [BoolVar(f"x_{i}") for i in range(4)]
    x = BinaryExpansion(vv)
    xx = Linear.of(x)

    v1 = ReLU(xx - 7)
    v2 = ReLU(xx - 8)
    # Full Processing ( 0 < Lin[Bin[x_0, x_1, x_2, x_3] - 8] < inf , {} ) , [RangeCondition]
    v = v1 - v2
    v = v.simplify2()

    # TODO: improve union \ intersection of conditions

    assert v == (x.coefficients[3] << 1)

    v = 8 * v
    u = xx - v

    result = BinaryExpansion(vv[:3])
    assert u == result


def test_ranges_over_conditional_function():
    x = Linear.of(Var("x"))

    # absolute value of x:
    func = ConditionalFunction([
        (x<0, -x),
        (x>=0, x)
    ])

    condition1 = (func < -10).simplify2()
    assert condition1 is FalseCondition

    condition1 = (func > -10).simplify2()
    assert condition1 is TrueCondition

    condition1 = (func < 10).simplify2()
    condition2 = (-10 < x) & (x < 10)
    assert condition1 == condition2

    condition1 = (func > 10).simplify2()
    condition2 = (x < -10) | (10 < x)
    assert condition1 == condition2

    condition1 = (func << 10).simplify2()
    condition2 = (x << -10) | (x << 10)
    assert condition1 == condition2


def test_output_promise():
    dummy_cond = [DummyCondition(values={i}) for i in range(2)]
    dummy_func = [DummyMap(i) for i in range(2)]

    cond_func = ConditionalFunction([
        (dummy_cond[i], dummy_func[i]) for i in range(2)
    ])
    assert not is_integral.compute(cond_func, simplifier_context)

    # TODO: Promises right now can update the function. Think to make it frozen and instead create a new function,
    #       since it might already "know" that it doesn't have an output promise, and composition with it might fail
    #       as well, and changing it after creation might be invisible to the process later.

    # One region with the output promise is not enough
    dummy_cond = [DummyCondition(values={i}) for i in range(2)]
    dummy_func = [DummyMap(i) for i in range(2)]

    dummy_func[0] = DummyMap(0, output_properties={is_integral: True})

    cond_func = ConditionalFunction([
        (dummy_cond[i], dummy_func[i]) for i in range(2)
    ])
    assert not is_integral.compute(cond_func, simplifier_context)

    # All region have the output promise
    dummy_cond = [DummyCondition(values={i}) for i in range(2)]
    dummy_func = [DummyMap(i) for i in range(2)]

    dummy_func[0] = DummyMap(0, output_properties={is_integral: True})
    dummy_func[1] = DummyMap(1, output_properties={is_integral: True})

    cond_func = ConditionalFunction([
        (dummy_cond[i], dummy_func[i]) for i in range(2)
    ])
    cond_func = cond_func.simplify2()
    assert is_integral.compute(cond_func, simplifier_context)


def test_mul_generation():
    x = Linear.of(Var("x"))

    func1 = (3 <= x) * x

    func2 = ConditionalFunction([
        (3<=x, x),
        (x<3, 0)
    ])

    assert func1 == func2
