from typing import List, Tuple

import pytest
from mapping_field.mapping_field import Var, NamedFunc, Func, MapElement


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var._instances = {}
    NamedFunc._instances = {}

class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([])
        self.value = value

    def to_string(self, vars_str_list: List[str]):
        return f'DummyMap({self.value})'

    def __eq__(self, other):
        return isinstance(other, DummyMap) and other.value == self.value
    
    def add(self, other):
        return super().add(other)

class ImprovedDummyMap(MapElement):
    def __init__(self, value: Tuple =(0,)):
        super().__init__([])
        self.value = value

    def to_string(self, vars_str_list: List[str]):
        return f'ImprovedDummyMap({self.value})'

    def __eq__(self, other):
        if isinstance(other, DummyMap):
            return len(self.value) == 1 and self.value[0] == other.value
        return isinstance(other, ImprovedDummyMap) and set(self.value) == set(other.value)

    def add(self, other):
        if isinstance(other, DummyMap):
            return ImprovedDummyMap(self.value + (other.value,))
        if isinstance(other, ImprovedDummyMap):
            return ImprovedDummyMap(self.value + other.value)
        return super().add(other)

    def mul(self, other):
        return self.add(other)

def test_addition_commutative_choice():
    elem1 = DummyMap(0)
    elem2 = ImprovedDummyMap((1,))

    addition1 = elem1 + elem2
    addition2 = elem2 + elem1
    assert addition1 == addition2

def test_multiplication_commutative_choice():
    elem1 = DummyMap(0)
    elem2 = ImprovedDummyMap((1,))

    multiplication1 = elem1 * elem2
    multiplication2 = elem2 * elem1
    assert multiplication1 == multiplication2

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
    x = DummyMap(0)
    assert x + 0 == x
    assert 0 + x == x


def test_zero_subtraction():
    x = DummyMap(0)
    assert x - 0 == x
    assert str(0 - x) == '(-DummyMap(0))'


def test_zero_multiplication():
    x = DummyMap(0)
    assert x * 0 == 0
    assert 0 * x == 0


def test_one_multiplication():
    x = DummyMap(0)
    assert x * 1 == x
    assert 1 * x == x


def test_zero_division():
    x = DummyMap(0)
    assert 0 / x == 0
    with pytest.raises(Exception):
        y = x / 0


def test_one_division():
    x = DummyMap(0)
    assert x / 1 == x


# ----------------- General Addition Rules -----------------

def test_neg_distributive():

    x, y, z = DummyMap(0), DummyMap(1), DummyMap(2)
    assert (-(-x)) == x
    assert str(-(x-y)) == '(DummyMap(1)-DummyMap(0))'


def test_addition_rules():
    x, y = DummyMap(0), DummyMap(1)
    assert str(x+y) == '(DummyMap(0)+DummyMap(1))'
    assert str(x+(-y)) == '(DummyMap(0)-DummyMap(1))'
    assert str((-x)+y) == '(DummyMap(1)-DummyMap(0))'
    assert str((-x)+(-y)) == '(-(DummyMap(0)+DummyMap(1)))'


def test_subtraction_rules():
    x, y = DummyMap(0), DummyMap(1)
    assert str(x-y) == '(DummyMap(0)-DummyMap(1))'
    assert str(x-(-y)) == '(DummyMap(0)+DummyMap(1))'
    assert str((-x)-y) == '(-(DummyMap(1)+DummyMap(0)))'  # TODO: 0,1 switched positions! Need a better way to compare.
    assert str((-x)-(-y)) == '(DummyMap(1)-DummyMap(0))'


# ----------------- General Multiplication Rules -----------------

def test_multiplication_rules():
    x, y, z = Var('x'), Var('y'), Var('z')
    f = Func('f')(x, y)
    g = Func('g')(z)
    h = Func('h')(y, z)

    assert str(((-x)/y) / (g*(-h)/(x * (-f)))) == '(-( (x*(x*f(x,y)))/(y*(g(z)*h(y,z))) ))'
