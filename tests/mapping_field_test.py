import pytest
from mapping_field.mapping_field import Var, Func


def test_associativity():
    x = Var('x')
    y = Var('y')
    z = Var('z')

    f = Func('f')(x)

    # lhs = y*f(x)
    # rhs = y*x - f(1/x)
    assert False

