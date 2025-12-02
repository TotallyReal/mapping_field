import pytest

from mapping_field.conditions import FalseCondition, TrueCondition
from mapping_field.mapping_field import InvalidInput, simplifier_context, Var
from mapping_field.property_engines import is_integral, is_condition
from mapping_field.tests.utils import DummyMap


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

    result = dummy1 + dummy2
    assert is_integral.compute(result, simplifier_context) is None
    result = dummy1 - dummy2
    assert is_integral.compute(result, simplifier_context) is None
    result = dummy1 * dummy2
    assert is_integral.compute(result, simplifier_context) is None

    simplifier_context.set_property(dummy1, is_integral, True)
    simplifier_context.set_property(dummy2, is_integral, True)

    result = dummy1 + dummy2
    assert is_integral.compute(result, simplifier_context) is True
    result = dummy1 - dummy2
    assert is_integral.compute(result, simplifier_context) is True
    result = dummy1 * dummy2
    assert is_integral.compute(result, simplifier_context) is True
