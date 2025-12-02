import pytest

from mapping_field.conditions import FalseCondition, TrueCondition
from mapping_field.log_utils.tree_loggers import TreeLogger, blue
from mapping_field.mapping_field import InvalidInput, Var, simplifier_context
from mapping_field.property_engines import is_condition, is_integral
from mapping_field.ranged_condition import IntervalRange, in_range
from mapping_field.tests.utils import DummyMap

simplify_logger = TreeLogger(__name__)

def test_int_var_promise():
    x = Var("x", output_properties={is_integral: True})

    assigned = x(3)
    with pytest.raises(InvalidInput):
        assigned = x(3.5)


def test_bool_var_promise():
    x = Var("x", output_properties={is_condition: True})

    assigned = x(0)
    assigned = x(1)
    assigned = x(TrueCondition)
    assigned = x(FalseCondition)

    with pytest.raises(InvalidInput):
        assigned = x(2)
    with pytest.raises(InvalidInput):
        assigned = x(0.5)


def test_integral_arithmetic():
    dummy1, dummy2 = DummyMap(1), DummyMap(2)

    result = -dummy1
    assert is_integral.compute(result, simplifier_context) is None
    result = dummy1 + dummy2
    assert is_integral.compute(result, simplifier_context) is None
    result = dummy1 - dummy2
    assert is_integral.compute(result, simplifier_context) is None
    result = dummy1 * dummy2
    assert is_integral.compute(result, simplifier_context) is None

    simplify_logger.log(blue("Adding Integral promise"))
    dummy1 = DummyMap(1, output_properties={is_integral: True})
    dummy2 = DummyMap(2, output_properties={is_integral: True})

    result = -dummy1
    assert is_integral.compute(result, simplifier_context) is True
    result = dummy1 + dummy2
    assert is_integral.compute(result, simplifier_context) is True
    result = dummy1 - dummy2
    assert is_integral.compute(result, simplifier_context) is True
    result = dummy1 * dummy2
    assert is_integral.compute(result, simplifier_context) is True

def test_integral_arithmetic_complex():
    dummy = [DummyMap(i, output_properties={is_integral: True}) for i in range(5)]

    result = (dummy[0] - dummy[1] + 3*dummy[2]) * (dummy[3] - dummy[4]) * dummy[0]
    assert is_integral.compute(result, simplifier_context) is True

def test_condition_implications():
    dummy = DummyMap(1, output_properties={is_condition: True})

    assert is_integral.compute(dummy, simplifier_context) is True
    assert in_range.compute(dummy, simplifier_context) == IntervalRange[0,1]