import pytest

from mapping_field.new_code.conditions import FalseCondition, TrueCondition
from mapping_field.new_code.mapping_field import InvalidInput, NamedFunc, Var
from mapping_field.new_code.promises import IntVar, IsIntegral
from mapping_field.new_code.ranged_condition import BoolVar
from mapping_field.new_code.tests.utils import DummyMap


def test_int_var_promise():
    x = IntVar('x')

    assigned = x(3)
    with pytest.raises(InvalidInput):
        assigned = x(3.5)

def test_bool_var_promise():
    x = BoolVar('x')

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
    assert result.has_promise(IsIntegral) is None
    result = dummy1 - dummy2
    assert result.has_promise(IsIntegral) is None
    result = dummy1 * dummy2
    assert result.has_promise(IsIntegral) is None

    dummy1.promises.add_promise(IsIntegral)
    dummy2.promises.add_promise(IsIntegral)

    result = dummy1 + dummy2
    assert result.has_promise(IsIntegral) is True
    result = dummy1 - dummy2
    assert result.has_promise(IsIntegral) is True
    result = dummy1 * dummy2
    assert result.has_promise(IsIntegral) is True