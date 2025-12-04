import logging

from mapping_field.binary_expansion import BinaryExpansion
from mapping_field.bool_vars import BoolVar
from mapping_field.conditional_function import ConditionalFunction, ReLU
from mapping_field.conditions import (
    FalseCondition, IntersectionCondition, TrueCondition, UnionCondition,
)
from mapping_field.log_utils.tree_loggers import TreeLogger, blue
from mapping_field.mapping_field import MapElementConstant, Var, simplifier_context
from mapping_field.property_engines import is_integral
from mapping_field.ranged_condition import IntervalRange, RangeCondition, in_range
from mapping_field.tests.utils import DummyCondition, DummyConditionOn, DummyMap

simplify_logger = TreeLogger(__name__)
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
        (dummy0 >= 0, 5),
    ]
    func = ConditionalFunction(regions)
    assert str(func) == "[ DummyMap(0)<0 -> x  ;  0<=DummyMap(0) -> 5 ]"

    # Changing the region list should not change the function
    regions[0] = (dummy0 < 1, 3)
    regions[1] = (dummy0 >=1, 2*x)
    region_changed_func = ConditionalFunction(regions)

    assert str(func) == "[ DummyMap(0)<0 -> x  ;  0<=DummyMap(0) -> 5 ]"
    assert str(region_changed_func) == "[ DummyMap(0)<1 -> 3  ;  1<=DummyMap(0) -> (2*x) ]"

    # Calling the function
    dummy1 = DummyMap(1)
    assigned_func = func({x: dummy1})

    assert str(assigned_func) == "[ DummyMap(0)<0 -> DummyMap(1)  ;  0<=DummyMap(0) -> 5 ]"
    # Some indication that func is frozen
    assert str(func) == "[ DummyMap(0)<0 -> x  ;  0<=DummyMap(0) -> 5 ]"


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


def test_equality_to_self():
    dummy_conditions = [DummyConditionOn(set_size=3, values=i) for i in range(3)]
    dummy = DummyMap(0)

    cond_func = ConditionalFunction([
        (dummy_conditions[0], dummy),
        (dummy_conditions[1], 1),
        (dummy_conditions[2], 2),
    ])

    assert cond_func == cond_func

    cond_func_same = ConditionalFunction([
        (dummy_conditions[0], dummy),
        (dummy_conditions[1], 1),
        (dummy_conditions[2], 2),
    ])

    assert cond_func == cond_func_same



def test_equality_to_standard_function():
    dummy_conditions = [DummyConditionOn(set_size=3, values=i) for i in range(3)]
    dummy = DummyMap(0)

    cond_func = ConditionalFunction([
        (dummy_conditions[0], dummy),
        (dummy_conditions[1], dummy),
        (dummy_conditions[2], dummy),
    ])

    assert cond_func.simplify() is dummy

    cond_func = ConditionalFunction([
        (dummy_conditions[0], dummy),
        (dummy_conditions[1], dummy),
        (dummy_conditions[2], 0),
    ])

    assert cond_func.simplify() != dummy


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

    result = result.simplify()

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
    cond_func = cond_func.simplify()

    simplified_version = ConditionalFunction([
        (dummy_cond[0] | dummy_cond[2], dummy_func[0]),
        (dummy_cond[1], dummy_func[1]),
    ])

    assert cond_func == simplified_version

    # Combine regions with assignments
    x = Var("x", output_properties={in_range: IntervalRange[0,10]})

    cond_func = ConditionalFunction([
        ( (0<=x) & (x<10), x + 3),
        (x << 10, MapElementConstant(13)),
    ])

    cond_func = cond_func.simplify()

    assert cond_func == x + 3


def test_linear_ranged_condition_subtraction():
    vv = [BoolVar(f"x_{i}") for i in range(4)]
    x = BinaryExpansion(vv)

    v1 = ReLU(x - 7)
    v2 = ReLU(x - 8)
    # Full Processing ( 0 < Lin[Bin[x_0, x_1, x_2, x_3] - 8] < inf , {} ) , [RangeCondition]
    v = v1 - v2
    v = v.simplify()

    # TODO: improve union \ intersection of conditions

    assert v == (x.coefficients[3] << 1)

    v = 8 * v
    u = x - v

    result = BinaryExpansion(vv[:3])
    assert u == result


def test_ranges_over_conditional_function():
    x = Var("x")

    # absolute value of x:
    func = ConditionalFunction([
        (x<0, -x),
        (x>=0, x)
    ])

    condition1 = (func < -10).simplify()
    assert condition1 is FalseCondition

    condition1 = (func > -10).simplify()
    assert condition1 is TrueCondition

    condition1 = (func < 10).simplify()
    condition2 = (-10 < x) & (x < 10)
    assert condition1 == condition2

    condition1 = (func > 10).simplify()
    condition2 = (x < -10) | (10 < x)
    assert condition1 == condition2

    condition1 = (func << 10).simplify()
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
    cond_func = cond_func.simplify()
    assert is_integral.compute(cond_func, simplifier_context)


def test_mul_generation():
    x = Var("x")

    func1 = (3 <= x) * x

    func2 = ConditionalFunction([
        (3<=x, x),
        (x<3, 0)
    ])

    assert func1 == func2

def test_sum_of_conditions():
    n = 2
    simplify_logger.tree.max_log_count = -1
    x = [Var(f'x_{i}') for i in range(n)]
    # simplify_logger.tree.set_active(False)
    elem = sum([x[i]<<2*i for i in range(n)], 1-n)
    elem = ReLU(elem)
    # simplify_logger.tree.set_active(True)
    cond1 = elem.simplify()
    cond2 = IntersectionCondition([x[i]<<2*i for i in range(n)])
    assert cond1 == cond2

class TestConditionalFunctionSimplifiers:

    def test_single_simplifier(self):
        cond = DummyCondition()
        dummy = DummyMap()

        element = ConditionalFunction([
            (UnionCondition([cond, ~cond]), dummy)
        ])

        element = element.simplify()
        assert element is dummy

    def test_combining_regions(self):
        # TODO: once removing MapProcessor, make better\more tests
        x, y = BoolVar("x"), BoolVar("y")

        func = ConditionalFunction([
                ((x << 0) & (y << 0), MapElementConstant(0)),
                ((x << 1) & (y << 0), MapElementConstant(1)),
                ((y << 1), x),
        ])

        func = func.simplify()
        assert func is x

    def test_nested_simplifier(self):
        cond_outer = DummyCondition(0)
        cond_inner = DummyCondition(1)
        dummy = [DummyMap(i) for i in range(3)]

        func_inner = ConditionalFunction([
                (cond_inner, dummy[0]),
                (~cond_inner, dummy[1]),
        ])

        func_outer = ConditionalFunction([
                (cond_outer, func_inner),
                (~cond_outer, dummy[2]),
        ])

        result = ConditionalFunction([
                (  cond_inner  & cond_outer, dummy[0]),
                ((~cond_inner) & cond_outer, dummy[1]),
                (~cond_outer, dummy[2]),
        ])

        simplify_logger.log(blue("simplifying"))
        func_outer = func_outer.simplify()

        simplify_logger.log(blue("Equating"))
        assert func_outer == result

    def test_double_constant_to_single_region_simplifier(self):
        cond = DummyCondition()
        func = ConditionalFunction([
            (cond, MapElementConstant(5)),
            (~cond, MapElementConstant(8)),
        ])
        func = func.simplify()

        version1 = 5 + (~cond) *MapElementConstant(3)
        version2 = 8 + cond * MapElementConstant(-3)

        assert func == version1 or func == version2
