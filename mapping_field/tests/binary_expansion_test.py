
from mapping_field.binary_expansion import BinaryExpansion
from mapping_field.mapping_field import Var
from mapping_field.bool_vars import BoolVar


def test_simple_construction():
    x, y = BoolVar("x"), BoolVar("y")
    func = BinaryExpansion([x, y, 0, 1])


def test_post_generation_independence():
    v: list[Var | int] = [BoolVar(f"v_{i}") for i in range(4)]

    v_copy = v.copy()
    func = BinaryExpansion(v_copy)

    assert str(func) == "Bin[v_0, v_1, v_2, v_3]"

    # Changing the input list v_copy should not change func
    v_copy[1] = 0
    v_copy[2] = 0
    func00 = BinaryExpansion(v_copy)

    assert func.simplify() != func00.simplify()
    assert str(func) == "Bin[v_0, v_1, v_2, v_3]"
    assert str(func00) == "Bin[v_0, 0, 0, v_3]"

    # Calling the function
    assigned = func({v[1]: 0, v[2]: 0})

    assert assigned.simplify() == func00.simplify()
    assert str(assigned) == "Bin[v_0, 0, 0, v_3]"
    # Some indication that func is frozen
    assert str(func) == "Bin[v_0, v_1, v_2, v_3]"


def test_equality_constant():
    x1 = BinaryExpansion([1, 0, 1, 1]).simplify()
    assert x1 == 13


def test_equality():
    v = BoolVar("v1")
    x1 = BinaryExpansion([v, 0, 1]).simplify()
    x2 = BinaryExpansion([v, 0, 1]).simplify()
    assert x1 == x2

    x2 = BinaryExpansion([v, 0, 1, 0, 0]).simplify()
    assert x1 == x2
    assert x2 == x1


def test_constant_split():
    x = BinaryExpansion([BoolVar("v1"), 0, 1])
    constant = 4
    pure = BinaryExpansion([BoolVar("v1")])
    assert x.split_constant() == (pure, constant)

    x = BinaryExpansion([1, 0, 1])
    constant = 5
    pure = None
    assert x.split_constant() == (pure, constant)

    x = BinaryExpansion([BoolVar("v1"), 0, BoolVar("v2")])
    constant = 0
    pure = BinaryExpansion([BoolVar("v1"), 0, BoolVar("v2")])
    assert x.split_constant() == (pure, constant)


def addition_test(x, y, x_plus_y):
    result = x + y
    assert result == x_plus_y.simplify()
    assert result - x == y.simplify()
    assert result - y == x.simplify()


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

def test_arithmetic_construction():
    v1 = BoolVar("v1")
    v2 = BoolVar("v2")

    result = v1 + 2 * v2
    bin_expa = BinaryExpansion([v1, v2])
    assert result == bin_expa

    result = BinaryExpansion.of(1 + 4 * result)
    bin_expa = BinaryExpansion([1, 0, v1, v2])
    assert result == bin_expa


def test_arithmetic():
    v1 = BoolVar("v1")
    v2 = BoolVar("v2")

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

    addition_test(
        BinaryExpansion([1, 1]),
        BinaryExpansion([1, v1]),
        BinaryExpansion([0, v1, 1])
    )


def test_linear_of_binary_expansion():
    xx = [BoolVar(f"x_{i}") for i in range(4)]
    full_bin = BinaryExpansion(xx)

    v1 = BinaryExpansion([xx[0], xx[1]])
    v2 = BinaryExpansion([xx[2], xx[3]])
    result = v1 + 4 * v2
    assert result == full_bin

    v1 = BinaryExpansion([xx[0], xx[1], xx[2]])
    result = v1 + 8 * xx[3]
    assert result == full_bin

    v1 = BinaryExpansion([xx[1], xx[2], xx[3]])
    result = 2 * v1 + xx[0]
    assert result == full_bin


def test_shift():
    v1 = BoolVar("v1")
    v2 = BoolVar("v2")

    x1 = BinaryExpansion([v1, v2, 0, 1, 1])
    x2 = BinaryExpansion([0, 0, v1, v2, 0, 1, 1])
    x3 = BinaryExpansion([0, 0, 0, 0, v1, v2, 0, 1, 1])

    assert x1.shift(+2) == x2
    assert x1.shift(+4) == x3

    assert x2.shift(-2) == x1
    assert x2.shift(+2) == x3

    assert x3.shift(-4) == x1
    assert x3.shift(-2) == x2


def test_simplify_range_condition():
    v = [BoolVar(f"v_{i}") for i in range(4)]
    x = BinaryExpansion(v)  # A number in [0,16)

    condition1 = ((3 <= x) & (x < 100)).simplify()
    condition2 = ((3 <= x) & (x < 16)).simplify()
    assert condition1 == condition2

    x = BinaryExpansion([1] + v)  # A number in [1,32), but only odd numbers

    condition1 = ((0 <= x) & (x < 10)).simplify()
    condition2 = ((1 <= x) & (x < 10)).simplify()
    assert condition1 == condition2

    # Only odd number in [16, 19) is 17, which is the only integer in [17,18)
    condition1 = ((16 <= x) & (x < 19)).simplify()
    condition2 = (x << 17).simplify()
    assert condition1 == condition2

    condition1 = ((17 <= x) & (x < 100)).simplify()
    condition2 = (v[3] << 1).simplify()
    assert condition1 == condition2


#
# def test_sum_of_bool():
#     x, y = BoolVar('x'), BoolVar('y')
#     condition = (0 < (x+y)-1)
#     condition = condition.simplify()
#
#     result = (x << 1) & (y << 1)
#
#     assert condition == result
#
