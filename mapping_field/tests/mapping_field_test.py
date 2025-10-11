from typing import List, Optional

import pytest

from mapping_field import MapElementConstant
from mapping_field.mapping_field import Var, NamedFunc, Func, CompositionFunction, MapElementFromFunction, MapElement, \
    VarDict


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var.clear_vars()
    NamedFunc.clear_vars()

class DummyMap(MapElement):

    def __init__(self, value: int):
        super().__init__([])
        self.value = value

    def to_string(self, vars_str_list: List[str]):
        return f'DummyMap({self.value})'


# ----------------- var tests -----------------

def test_var_double_generation():
    x1 = Var('x')
    x2 = Var('x')
    assert x1 == x2


def test_var_try_get():
    x = Var('x')
    assert Var.try_get('x') == x
    assert Var.try_get('z') is None


def test_var_string():
    x = Var('x')
    assert str(x) == 'x'
    assert repr(x) == 'x'


def test_var_assignment():
    dummy = DummyMap(0)
    x = Var('x')
    assert x(dummy) == dummy


# ----------------- named function tests -----------------

def test_named_function_double_generation():
    x, y = Var('x'), Var('y')
    f = NamedFunc('f', [x, y])

    # Trying to create the same function with the same exact variables is fine:
    g = NamedFunc('f', [x, y])

    # different variables will raise an exception
    with pytest.raises(Exception) as ex:
        g = NamedFunc('f', [y, x])


def test_named_function_try_get():
    x, y = Var('x'), Var('y')
    f = NamedFunc('f', [x, y])
    assert NamedFunc.try_get('f') == f
    assert NamedFunc.try_get('g') is None


def test_name_function_string():
    x, y = Var('x'), Var('y')
    f = NamedFunc('f', [x, y])
    assert str(f) == 'f(x,y)'


def test_name_function_assignment():
    pass


# ----------------- named function generation tests -----------------

def test_named_function_generation():
    x, y = Var('x'), Var('y')
    f = Func('f')(x, y)
    assert str(f) == 'f(x,y)'


# ----------------- composition test -----------------

def test_composition_top_function():
    x = Var('x')
    f = Func('f')(x)
    g = Func('g')(x)
    h = Func('h')(x)

    comp_function: CompositionFunction = f(g(h))
    assert comp_function.function == f

    fg = f(g)
    comp_function: CompositionFunction = fg(h)
    assert comp_function.function == f


# ----------------- simplify test -----------------

def test_simplify():
    addition = MapElementFromFunction(name='Add', function=lambda a, b: a+b)
    assert str(addition(2, 3)) == '5'
    assert str(addition(2, 3, simplify=False)) == 'Add(2,3)'


class DummyMapWithVar(MapElement):
    def __init__(self, value=0):
        self.x = Var('x')
        super().__init__([self.x], f'DummyMap_{value}')
        self.value = value

    def __eq__(self, other):
        return isinstance(other, DummyMap) and other.value == self.value

    # Override when needed
    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        return MapElementConstant.zero if var_dict[self.x] == 0 else None


class SpecialDummyVar(MapElement):
    def __init__(self, value=0):
        super().__init__([], f'SpecialDummyVar_{value}')
        self.value = value

    def _simplify_caller_function2(self, function: MapElement, position: int, var_dict: VarDict) -> Optional[MapElement]:
        if isinstance(function, DummyMapWithVar) and self.value == function.value:
            return MapElementConstant.one
        return None


def test_simplify2():
    dummy = DummyMapWithVar()
    assert str(dummy) == 'DummyMap_0(x)'

    # simplification from caller function

    assigned = dummy(1)
    assert str(assigned) == 'DummyMap_0(1)'

    assigned = dummy(0, simplify=False)
    assert str(assigned) == 'DummyMap_0(0)'

    assigned = dummy(0)
    assert assigned == 0

    # simplification from variable behaviour

    assigned = dummy(SpecialDummyVar(1))
    assert str(assigned) == 'DummyMap_0(SpecialDummyVar_1)'

    assigned = dummy(SpecialDummyVar(0), simplify=False)
    assert str(assigned) == 'DummyMap_0(SpecialDummyVar_0)'

    assigned = dummy(SpecialDummyVar(0))
    assert assigned == 1




