import pytest

from mapping_field.conditions import FalseCondition, TrueCondition
from mapping_field.mapping_field import MapElementConstant, SimplifierContext, Var
from mapping_field.property_engines import is_condition, is_integral
from mapping_field.tests.utils import DummyMap

pytestmark = pytest.mark.order(1)

def test_constant_is_condition():
    context = SimplifierContext()

    elem = MapElementConstant(0)
    assert is_condition.compute(elem, context)

    elem = MapElementConstant(1)
    assert is_condition.compute(elem, context)

    elem = MapElementConstant(-1)
    assert not is_condition.compute(elem, context)

    elem = MapElementConstant(5.5)
    assert not is_condition.compute(elem, context)


def test_binary_is_condition():
    context = SimplifierContext()

    assert is_condition.compute(TrueCondition, context)
    assert is_condition.compute(FalseCondition, context)


def test_constant_integrality():
    context = SimplifierContext()

    elem = MapElementConstant(5)
    assert is_integral.compute(elem, context)

    elem = MapElementConstant(5.5)
    assert not is_integral.compute(elem, context)


def test_integral_preserving():
    dummy1, dummy2 = DummyMap(1), DummyMap(2)
    context = SimplifierContext()

    addition = dummy1 + dummy2

    assert not is_integral.compute(addition, context)

    context.set_property(dummy1, is_integral, True)
    assert not is_integral.compute(addition, context)

    context.set_property(dummy2, is_integral, True)
    assert is_integral.compute(addition, context)


def test_condition_to_integral():
    context = SimplifierContext()

    dummy = DummyMap(0)

    assert not is_integral.compute(dummy, context)

    context.set_property(dummy, is_condition, True)

    assert is_integral.compute(dummy, context)