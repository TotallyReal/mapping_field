import pytest

from mapping_field.conditions import FalseCondition, TrueCondition
from mapping_field.mapping_field import MapElementConstant, SimplifierContext, simplifier_context
from mapping_field.property_engines import is_condition, is_integral
from mapping_field.tests.utils import DummyMap


@pytest.fixture
def context() -> SimplifierContext:
    return simplifier_context

#       ╭─────────────────────────────────────────────────╮
#       │                   Condition                     │
#       ╰─────────────────────────────────────────────────╯

def test_constant_is_condition(context):
    elem = MapElementConstant(0)
    assert is_condition.compute(elem, context)

    elem = MapElementConstant(1)
    assert is_condition.compute(elem, context)

    elem = MapElementConstant(-1)
    assert not is_condition.compute(elem, context)

    elem = MapElementConstant(5.5)
    assert not is_condition.compute(elem, context)


def test_binary_is_condition(context):

    assert is_condition.compute(TrueCondition, context)
    assert is_condition.compute(FalseCondition, context)

#       ╭─────────────────────────────────────────────────╮
#       │                    Integral                     │
#       ╰─────────────────────────────────────────────────╯

def test_constant_integrality(context):

    elem = MapElementConstant(5)
    assert is_integral.compute(elem, context)

    elem = MapElementConstant(5.5)
    assert not is_integral.compute(elem, context)


def test_integral_preserving(context):
    dummy1, dummy2 = DummyMap(1), DummyMap(2)

    addition = dummy1 + dummy2

    assert not is_integral.compute(addition, context)

    context.set_property(dummy1, is_integral, True)
    assert not is_integral.compute(addition, context)

    context.set_property(dummy2, is_integral, True)
    assert is_integral.compute(addition, context)


def test_condition_to_integral(context):

    dummy = DummyMap(0)
    assert not is_integral.compute(dummy, context)

    context.set_property(dummy, is_condition, True)
    assert is_integral.compute(dummy, context)