from typing import Optional, Tuple

import pytest

from mapping_field.arithmetics import _as_combination
from mapping_field.mapping_field import Func, MapElement, Var, VarDict
from mapping_field.tests.utils import DummyMap


class ImprovedDummyMap(MapElement):
    def __init__(self, value: Tuple = (0,)):
        super().__init__([], f"ImprovedDummyMap({value})")
        self.value = value

    def __eq__(self, other):
        if isinstance(other, DummyMap):
            return len(self.value) == 1 and self.value[0] == other.value
        return isinstance(other, ImprovedDummyMap) and set(self.value) == set(other.value)

    def _simplify_caller_function2(
        self, function: MapElement, position: int, var_dict: VarDict
    ) -> Optional[MapElement]:
        elem2 = var_dict[function.vars[1 - position]]
        if function in (MapElement.addition, MapElement.multiplication):
            if isinstance(elem2, DummyMap):
                return ImprovedDummyMap(self.value + (elem2.value,))
            if isinstance(elem2, ImprovedDummyMap):
                return ImprovedDummyMap(self.value + elem2.value)
        return None


def test_arithmetic_premade_methods():
    class DummyMapNeg(DummyMap):

        def neg(self) -> Optional[MapElement]:
            return DummyMap(-self.value) if self.value != 0 else None

    dummy = DummyMapNeg(0)
    assert str(-dummy) == "(-DummyMap(0))"

    dummy = DummyMapNeg(1)
    assert str(-dummy) == "DummyMap(-1)"


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
    x, y, z = Var("x"), Var("y"), Var("z")

    f = (x + y) * z + 5 - (2 * x * x)
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
    assert str(0 - x) == "(-DummyMap(0))"


def test_addition_inverse():
    x = DummyMap(0)
    assert x - x == 0
    assert x + (-x) == 0
    assert (-x) + x == 0


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
    assert str(-(x - y)) == "(DummyMap(1)-DummyMap(0))"


def test_addition_rules():
    x, y = DummyMap(0), DummyMap(1)
    assert str(x + y) == "(DummyMap(0)+DummyMap(1))"
    assert str(x + (-y)) == "(DummyMap(0)-DummyMap(1))"
    assert str((-x) + y) == "(DummyMap(1)-DummyMap(0))"
    assert str((-x) + (-y)) == "(-(DummyMap(0)+DummyMap(1)))"


def test_subtraction_rules():
    x, y = DummyMap(0), DummyMap(1)
    assert str(x - y) == "(DummyMap(0)-DummyMap(1))"
    assert str(x - (-y)) == "(DummyMap(0)+DummyMap(1))"

    # TODO: 0,1 switched positions! Need a better way to compare.
    assert str((-x) - y) == "(-(DummyMap(1)+DummyMap(0)))"
    assert str((-x) - (-y)) == "(DummyMap(1)-DummyMap(0))"


# ----------------- General Multiplication Rules -----------------


def test_multiplication_rules():
    x, y, z = Var("x"), Var("y"), Var("z")

    assert str(x * (-y)) == "(-(x*y))"
    assert str((-x) * y) == "(-(x*y))"
    assert str((-x) * (-y)) == "(x*y)"

    f = Func("f")(x, y)
    g = Func("g")(z)
    h = Func("h")(y, z)

    assert str(((-x) / y) / (g * (-h) / (x * (-f)))) == "(-( (x*(x*f(x,y)))/(y*(g(z)*h(y,z))) ))"


# ----------------- assignment -----------------


def test_simplification_after_assignment():
    simple_addition = ImprovedDummyMap((0,)) + ImprovedDummyMap((1,))
    result = ImprovedDummyMap((0, 1))
    assert simple_addition == result

    x = Var("x")
    y = ImprovedDummyMap((0,))
    assigned_addition = x + y

    assigned_addition = assigned_addition({x: ImprovedDummyMap((1,))})
    assigned_addition = assigned_addition.simplify2()
    assert assigned_addition == result


# ----------------- combination -----------------


def test_as_combination_trivial():
    dummy0 = DummyMap(0)

    a, elem_a, b, elem_b = _as_combination(dummy0)
    assert (a, elem_a, b) == (1, dummy0, 0)

    a, elem_a, b, elem_b = _as_combination(-dummy0)
    assert (a, elem_a, b) == (-1, dummy0, 0)


def test_as_combination_only_multiplication():
    dummy0 = DummyMap(0)

    a, elem_a, b, elem_b = _as_combination(3 * dummy0)
    assert (a, elem_a, b) == (3, dummy0, 0)

    a, elem_a, b, elem_b = _as_combination(dummy0 * 3)
    assert (a, elem_a, b) == (3, dummy0, 0)

    a, elem_a, b, elem_b = _as_combination((-3) * dummy0)
    assert (a, elem_a, b) == (-3, dummy0, 0)

    a, elem_a, b, elem_b = _as_combination(-(3 * dummy0))
    assert (a, elem_a, b) == (-3, dummy0, 0)


def test_as_combination_linear():
    dummy0 = DummyMap(0)

    a, elem_a, b, elem_b = _as_combination(3 * dummy0 + 4)
    assert (a, elem_a, b) == (3, dummy0, 4)

    a, elem_a, b, elem_b = _as_combination(4 + dummy0 * 3)
    assert (a, elem_a, b) == (3, dummy0, 4)

    a, elem_a, b, elem_b = _as_combination(3 * dummy0 - 4)
    assert (a, elem_a, b) == (3, dummy0, -4)

    a, elem_a, b, elem_b = _as_combination(4 - dummy0 * 3)
    assert (a, elem_a, b) == (-3, dummy0, 4)


def test_as_combination_full():
    dummy0 = DummyMap(0)
    dummy1 = DummyMap(1)

    a, elem_a, b, elem_b = _as_combination(3 * dummy0 + 4 * dummy1)
    assert (a, elem_a, b, elem_b) == (3, dummy0, 4, dummy1)
