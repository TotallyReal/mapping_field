import pytest
from mapping_field.mapping_field import Var, NamedFunc, Func


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var._instances = {}
    NamedFunc._instances = {}


# ----------------- test simple arithmetics -----------------


def test_simple_arithmetics():
    x, y, z = Var('x'), Var('y'), Var('z')

    f = (x+y)*z + 5 - (2*x*x)
    f.set_var_order([x, y, z])
    assert f(0, 0, 0) == 5
    assert f(1, 0, 0) == 3
    assert f(1, 2, 0) == 3
    assert f(1, 2, 3) == 12
    assert f(0, 2, 3) == 11
    assert f(0, 0, 3) == 5


# ----------------- 0 / 1 rules -----------------

# TODO: create TestMapElement
# TODO: not the best way to test with str(.), but it will do until I can compare functions

def test_zero_addition():
    x = Var('x')
    assert str(x+0) == 'x'
    assert str(0+x) == 'x'


def test_zero_subtraction():
    x = Var('x')
    assert str(x - 0) == 'x'
    assert str(0 - x) == '(-x)'


def test_zero_multiplication():
    x = Var('x')
    assert str(x * 0) == '0'
    assert str(0 * x) == '0'


def test_one_multiplication():
    x = Var('x')
    assert str(x * 1) == 'x'
    assert str(1 * x) == 'x'


def test_zero_division():
    x = Var('x')
    assert str(0 / x) == '0'
    with pytest.raises(Exception):
        assert str(x / 0) == '0'


def test_one_division():
    x = Var('x')
    assert str(x / 1) == 'x'


# ----------------- General Addition Rules -----------------

def test_neg_distributive():

    x, y, z = Var('x'), Var('y'), Var('z')
    assert str(-(-x)) == 'x'
    assert str(-(x-y)) == '(y-x)'


def test_addition_rules():
    x, y = Var('x'), Var('y')
    assert str(x+y) == '(x+y)'
    assert str(x+(-y)) == '(x-y)'
    assert str((-x)+y) == '(y-x)'
    assert str((-x)+(-y)) == '(-(x+y))'


def test_subtraction_rules():
    x, y = Var('x'), Var('y')
    assert str(x-y) == '(x-y)'
    assert str(x-(-y)) == '(x+y)'
    assert str((-x)-y) == '(-(y+x))'  # TODO: x,y switched positions! Need a better way to compare.
    assert str((-x)-(-y)) == '(y-x)'


# ----------------- General Multiplication Rules -----------------

def test_multiplication_rules():
    x, y, z = Var('x'), Var('y'), Var('z')
    f = Func('f')(x, y)
    g = Func('g')(z)
    h = Func('h')(y, z)

    assert str(((-x)/y) / (g*(-h)/(x * (-f)))) == '(-( (x*(x*f(x,y)))/(y*(g(z)*h(y,z))) ))'
