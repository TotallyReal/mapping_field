from mapping_field import MapElementConstant
from mapping_field.binary_expansion import BinaryExpansion, BoolVar


def test_equality_constant():
    x1 = BinaryExpansion([1, 0, 1, 1])
    assert x1 == 13
    assert 13 == x1

    x2 = MapElementConstant(13)
    assert x1 == x2
    assert x2 == x1

def test_equality():
    x1 = BinaryExpansion([BoolVar('v1'), 0, 1])
    x2 = BinaryExpansion([BoolVar('v1'), 0, 1])
    assert x1 == x2

    x1 = BinaryExpansion([BoolVar('v1'), 0, 1])
    x2 = BinaryExpansion([BoolVar('v1'), 0, 1, 0, 0])
    assert x1 == x2
    assert x2 == x1

def addition_test(x, y, x_plus_y):
    result = x + y
    assert result == x_plus_y
    assert result - x == y
    assert result - y == x

def test_arithmetic_constant():
    addition_test(
        BinaryExpansion([1, 0, 0, 1, 1, 1, 0, 1]),
        BinaryExpansion([0, 1, 0, 1, 1, 1, 0, 1]),
        BinaryExpansion([1, 1, 0, 0, 1, 1, 1, 0, 1])
    )

    # extra carry in the end
    addition_test(
        BinaryExpansion([1, 1, 0, 1]),
        BinaryExpansion([0, 1, 0]),
        BinaryExpansion([1, 0, 1, 1])
    )

def test_arithmetic():
    v1 = BoolVar('v1')
    v2 = BoolVar('v2')

    addition_test(
        BinaryExpansion([v1, 0, 1]),
        BinaryExpansion([0, v2, 1]),
        BinaryExpansion([v1, v2, 0, 1])
    )

    addition_test(
        BinaryExpansion([v1, 0, 1]),
        BinaryExpansion([0, v2, 0, 1]),
        BinaryExpansion([v1, v2, 1, 1])
    )

def test_shift():
    v1 = BoolVar('v1')
    v2 = BoolVar('v2')

    x1 = BinaryExpansion([v1, v2, 0, 1, 1])
    x2 = BinaryExpansion([0, 0, v1, v2, 0, 1, 1])
    x3 = BinaryExpansion([0, 0, 0, 0, v1, v2, 0, 1, 1])

    assert x1.shift(+2) == x2
    assert x1.shift(+4) == x3

    assert x2.shift(-2) == x1
    assert x2.shift(+2) == x3

    assert x3.shift(-4) == x1
    assert x3.shift(-2) == x2
