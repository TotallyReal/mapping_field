import pytest
from mapping_field.mapping_field import Var, NamedFunc, Func, CompositionFunction, MapElementFromFunction


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var._instances = {}
    NamedFunc._instances = {}


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
    pass


# ----------------- named function tests -----------------

def test_named_function_double_generation():
    x, y = Var('x'), Var('y')
    f = NamedFunc('f', [x, y])
    with pytest.raises(Exception) as ex:
        g = NamedFunc('f', [x, y])


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




# def test_associativity():
#     x = Var('x')
#     y = Var('y')
#     z = Var('z')
#
#     f = Func('f')(x)
#
#     # lhs = y*f(x)
#     # rhs = y*x - f(1/x)
#     assert False

