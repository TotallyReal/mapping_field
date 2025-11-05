import pytest

from mapping_field.new_code.conditions import TrueCondition, FalseCondition
from mapping_field.new_code.mapping_field import Var, NamedFunc, InvalidInput
from mapping_field.new_code.promises import IntVar, BoolVar


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var.clear_vars()
    NamedFunc.clear_vars()

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