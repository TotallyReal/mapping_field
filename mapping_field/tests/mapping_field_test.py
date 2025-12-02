
import pytest

from mapping_field.mapping_field import (
    CompositeElement, CompositeElementFromFunction, Func, MapElement, MapElementConstant, NamedFunc,
    SimplifierOutput, Var, ConflictingVariables, InvalidVariableOrder,
)
from mapping_field.tests.utils import DummyMap

# ----------------- var tests -----------------


def test_var_double_generation():
    x1 = Var("x")
    x2 = Var("x")
    assert x1 == x2


def test_same_name_variables():
    x1 = Var("x")
    x2 = Var("x")
    assert id(x1) != id(x2)

    with pytest.raises(ConflictingVariables):
        MapElement(variables=[x1, x2])


def test_set_var_order():
    x, y, z = Var("x"), Var("y"), Var("z")
    f = MapElement(variables=[x, y])

    assert f.vars == [x, y]

    f.set_var_order([y, x])
    assert f.vars == [y, x]

    with pytest.raises(ConflictingVariables):
        f.set_var_order([x, x])

    # the variables did not change
    assert f.vars == [y, x]

    with pytest.raises(InvalidVariableOrder):
        f.set_var_order([y, z])

    # the variables did not change
    assert f.vars == [y, x]


def test_var_string():
    x = Var("x")
    assert str(x) == "x"


def test_var_assignment():
    dummy = DummyMap(0)
    x = Var("x")
    assert x(dummy) is dummy
    assert x({x: dummy}) is dummy


# ----------------- named function tests -----------------


def test_named_function_double_generation():
    x, y = Var("x"), Var("y")
    f = NamedFunc("f", [x, y])

    # Trying to create the same function with the same exact variables is fine:
    g = NamedFunc("f", [x, y])

    # different variables will raise an exception
    with pytest.raises(Exception) as ex:
        g = NamedFunc("f", [y, x])


def test_named_function_try_get():
    x, y = Var("x"), Var("y")
    f = NamedFunc("f", [x, y])
    assert NamedFunc.try_get("f") == f
    assert NamedFunc.try_get("g") is None


def test_name_function_string():
    x, y = Var("x"), Var("y")
    f = NamedFunc("f", [x, y])
    assert str(f) == "f(x,y)"


def test_name_function_assignment():
    pass


# ----------------- named function generation tests -----------------


def test_named_function_generation():
    x, y = Var("x"), Var("y")
    f = Func("f")(x, y)
    assert str(f) == "f(x,y)"


# ----------------- composition test -----------------


# def test_composition_top_function():
#     x = Var("x")
#     f = Func("f")(x)
#     g = Func("g")(x)
#     h = Func("h")(x)
#
#     comp_function: CompositionFunction = f(g(h))
#     assert comp_function.function == f
#
#     fg = f(g)
#     comp_function: CompositionFunction = fg(h)
#     assert comp_function.function == f


# def test_assignment_using_composition():
#     x = Var("x")
#     dummy = DummyMap(0)
#     comp_function = CompositionFunction(x, [dummy])
#     assert str(x) == "x"
#     assert str(comp_function) == "DummyMap(0)"
#
#     assert comp_function.simplify2() is dummy


# ----------------- simplify test -----------------


def test_simplify():
    addition = CompositeElementFromFunction(name="Add", function=lambda a, b: a + b)
    assert str(addition(2, 3)) == "5"
    assert str(addition(2, 3, simplify=False)) == "Add(2,3)"


class DummyMapWithVar(CompositeElement):
    def __init__(self, value=0):
        self.x = Var("x")
        super().__init__(operands=[self.x], name=f"DummyMap_{value}")
        self.value = value

    def __eq__(self, other):
        return isinstance(other, DummyMap) and other.value == self.value

    __hash__ = MapElement.__hash__

    # Override when needed
    def _simplify_with_var_values2(self) -> SimplifierOutput:
        return MapElementConstant.zero if (self.operands[0] == 0) else None

    @staticmethod
    def _dummy_operand_simplifier(self) -> SimplifierOutput:
        operand = self.operands[0]
        if hasattr(operand, "_dummy_operand_op"):
            return operand._dummy_operand_op(self)
        return None

DummyMapWithVar.register_class_simplifier(DummyMapWithVar._dummy_operand_simplifier)


class SpecialDummyVar(MapElement):
    def __init__(self, value=0):
        super().__init__([], f"SpecialDummyVar_{value}")
        self.value = value

    def _dummy_operand_op(self, function: MapElement) -> SimplifierOutput:
        if isinstance(function, DummyMapWithVar) and self.value == function.value:
            return MapElementConstant.one
        return None


def test_simplify2():
    dummy = DummyMapWithVar()
    assert str(dummy) == "DummyMap_0(x)"

    # simplification from caller function

    assigned = dummy(1)
    assert str(assigned) == "DummyMap_0(1)"

    assigned = dummy(0, simplify=False)
    assert str(assigned) == "DummyMap_0(0)"

    assigned = dummy(0)
    assert assigned == 0

    # simplification from variable behaviour

    assigned = dummy(SpecialDummyVar(1))
    assert str(assigned) == "DummyMap_0(SpecialDummyVar_1)"

    assigned = dummy(SpecialDummyVar(0), simplify=False)
    assert str(assigned) == "DummyMap_0(SpecialDummyVar_0)"

    assigned = dummy(SpecialDummyVar(0))
    assert assigned == 1

# TODO: Once the new FunctionComposition class is ready, return this test
# def test_adding_element_simplifier():
#     x = Var("x")
#     f = Func("f")(x)
#     g = Func("g")(x)
#
#     dummy0 = DummyMap(0)
#     dummy1 = DummyMap(1)
#
#     assert str(f(dummy0)) == "f(DummyMap(0))"
#     assert f(dummy0) != 0
#     assert str(g(dummy0)) == "g(DummyMap(0))"
#     assert str(f(dummy1)) == "f(DummyMap(1))"
#
#     def f0_simplifier(var_dict: VarDict) -> Optional[MapElement]:
#         dummy_param = var_dict.get(x, None)
#         if isinstance(dummy_param, DummyMap) and dummy_param.value == 0:
#             return MapElementConstant.zero
#         return None
#
#     f.register_simplifier(f0_simplifier)
#
#     assert f(dummy0) == 0
#     assert str(g(dummy0)) == "g(DummyMap(0))"
#     assert str(f(dummy1)) == "f(DummyMap(1))"

# TODO: registering class simplifiers are not reset between tests.
#       find a solution to this problem

# def test_adding_class_simplifier():
#     x, y = Var("x"), Var("y")
#     f = Func("f")(x)
#     g = Func("g")(x, y)
#     h = Func("h")(y)
#
#     dummy0 = DummyMap(0)
#     dummy1 = DummyMap(1)
#
#     assert str(f(dummy0)) == "f(DummyMap(0))"
#     assert f(dummy0) != 0
#     assert str(g(dummy0, dummy1)) == "g(DummyMap(0),DummyMap(1))"
#     assert g(dummy0, dummy1) != 0
#     assert str(h(dummy0)) == "h(DummyMap(0))"
#     assert h(dummy0) != 0
#
#     def named_func_simplifier(map_elem: MapElement, var_dict: VarDict) -> Optional[MapElement]:
#         assert isinstance(map_elem, NamedFunc)  # should be automatic
#
#         dummy_param = var_dict.get(x, None)
#         if isinstance(dummy_param, DummyMap) and dummy_param.value == 0:
#             return MapElementConstant.zero
#
#         return None
#
#     NamedFunc.register_class_simplifier(named_func_simplifier)
#
#     assert f(dummy0) == 0
#     assert g(dummy0, dummy1) == 0
#     assert str(h(dummy0)) == "h(DummyMap(0))"
#     assert h(dummy0) != 0


def test_unique_01():
    assert MapElementConstant(0) is MapElementConstant.zero
    assert MapElementConstant(1) is MapElementConstant.one


def test_double_simplification():

    class SimplifiedDummyMap(DummyMap):
        def __init__(self):
            super().__init__()
            self.simplified_counter = 0

        def _simplify_with_var_values2(self) -> SimplifierOutput:
            self.simplified_counter += 1
            return MapElementConstant(5)

    dummy = SimplifiedDummyMap()

    assert dummy.simplified_counter == 0

    # First simplification:
    assert dummy.simplify2() == 5
    assert dummy.simplified_counter == 1

    # Second simplification:
    assert dummy != 5
    assert dummy.simplify2() == 5
    assert dummy.simplified_counter == 1  # did not go up to 2


def test_double_simplification_assignment():

    class SimplifiedDummyMap(DummyMap):
        def __init__(self):
            super().__init__()
            self.simplified_counter = 0

        def _simplify_with_var_values2(self) -> SimplifierOutput:
            self.simplified_counter += 1
            return None

    dummy = SimplifiedDummyMap()

    assert dummy.simplified_counter == 0

    dummy.simplify2()

    assert dummy.simplified_counter == 1

    function = MapElement.multiplication(MapElementConstant.one, dummy, simplify = False)
    function.simplify2()

    # The simplification process here is :
    #   1 * dummy   ->   dummy  ->   simplified
    # Since dummy was encountered in the middle, in previous versions in caused it to be simplified again.
    # Adding this test here to make sure it doesn't happen again
    assert dummy.simplified_counter == 1
