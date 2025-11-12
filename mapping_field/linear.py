import math

from typing import Dict, Optional, Union

from mapping_field.arithmetics import Add, BinaryCombination, Sub, _as_combination
from mapping_field.conditions import FalseCondition, TrueCondition
from mapping_field.log_utils.tree_loggers import TreeLogger, green
from mapping_field.mapping_field import (
    ExtElement, FuncDict, MapElement, MapElementConstant, Var, VarDict, get_var_values,
    params_to_maps,
)
from mapping_field.processors import ProcessFailureReason
from mapping_field.ranged_condition import RangeCondition
from mapping_field.serializable import DefaultSerializable

logger = TreeLogger(__name__)


class Linear(MapElement, DefaultSerializable):

    @staticmethod
    def of(elem: MapElement):
        value = elem.evaluate()
        if value is not None:
            return Linear(0, MapElementConstant.zero, value)

        if isinstance(elem, Linear):
            return elem

        a, elem_a, b, elem_b = _as_combination(elem)
        if b == 0 or elem_b is MapElementConstant.one:
            return Linear(a, elem_a, b)
        if a == 0 or elem_a is MapElementConstant.one:
            return Linear(b, elem_b, a)

        return Linear(1, elem, 0)

    def __init__(self, a: float, elem: MapElement, b: float):
        super().__init__(elem.vars)
        self.a = a
        self.b = b
        self.elem = elem

    def to_string(self, vars_to_str: Dict[Var, str]):
        a_str = f"{str(self.a)}*" if self.a != 1 else ""
        b_str = ""
        if self.b > 0:
            b_str = f" + {self.b}"
        if self.b < 0:
            b_str = f" - {-self.b}"
        return f"Lin[{a_str}{self.elem.to_string(vars_to_str)}{b_str}]"

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        elem = self.elem._call_with_dict(var_dict, func_dict)
        if elem is self.elem:
            return self
        return Linear(self.a, self.elem._call_with_dict(var_dict, func_dict), self.b)

    # <editor-fold desc=" ------------------------ Arithmetics ------------------------">

    def neg(self) -> Optional[MapElement]:
        return Linear(-self.a, self.elem, -self.b)

    # The Linear addition always tries to return a "simplified" Linear map.
    def add(self, other: MapElement) -> Optional[MapElement]:
        value = other.evaluate() if isinstance(other, MapElement) else other
        if isinstance(value, int):
            return Linear(self.a, self.elem, self.b + value)

        lin_other = Linear.of(other)

        a1, elem1, b1 = self.a, self.elem, self.b
        a2, elem2, b2 = lin_other.a, lin_other.elem, lin_other.b

        if elem1 == elem2:
            return Linear(a1 + a2, elem1, b1 + b2)

        if isinstance(a1, int) and isinstance(a2, int):
            gcd = math.gcd(a1, a2)

            bin_comb = BinaryCombination(a1 // gcd, elem1, a2 // gcd, elem2)
            result = bin_comb._simplify2()
            if result is not None and not isinstance(result, BinaryCombination):
                return Linear(gcd, result, b1 + b2)

        return super().add(other)

    def radd(self, other: MapElement) -> Optional[MapElement]:
        return self.add(other)

    def sub(self, other: MapElement) -> Optional[MapElement]:
        return self.add(-other)

    def rsub(self, other: MapElement) -> Optional[MapElement]:
        return (-self).add(other)

    def mul(self, other: MapElement) -> Optional[MapElement]:
        n = other.evaluate()
        if isinstance(n, int):
            return Linear(self.a * n, self.elem, self.b * n)
        return super().mul(other)

    def rmul(self, other: MapElement) -> Optional[MapElement]:
        return self.mul(other)

    def div(self, other: MapElement) -> Optional[MapElement]:
        if isinstance(other, MapElementConstant) and isinstance(other.elem, (int, float)):
            other = other.elem
        if isinstance(other, (int, float)):
            return Linear(self.a / other, self.elem, self.b / other)
        return super().div(other)

    # </editor-fold>

    def evaluate(self) -> Optional[ExtElement]:
        if self.a == 0:
            return self.b

        n = self.elem.evaluate()
        return None if (n is None) else self.a * n + self.b

    @params_to_maps
    def __eq__(self, other: MapElement):
        if other is MapElementConstant.zero:
            return self.evaluate() == 0
        return (self - Linear.of(other)).evaluate() == 0

    # <editor-fold desc=" ------------------------ Simplifiers ------------------------ ">

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        if self.a == 0:
            return MapElementConstant(self.b)

        elem = self.elem._simplify2(var_dict)
        return None if elem is None else Linear(self.a, elem, self.b)

    def _transform_linear(element: MapElement, var_dict: VarDict) -> Optional[Union[MapElement, ProcessFailureReason]]:
        assert isinstance(element, Linear)
        if isinstance(element.elem, Linear):
            a, b = element.a, element.b
            a_, b_ = element.elem.a, element.elem.b
            return Linear(int(a * a_), element.elem.elem, int(a * b_ + b))
        return ProcessFailureReason("Inner element is not Linear", trivial=True)

    def _evaluate_simplifier(self, var_dict: VarDict) -> Optional[MapElement]:
        value = self.evaluate()
        return MapElementConstant(value) if value is not None else None

    @staticmethod
    def _transform_range(element: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        """
        If the range is over a Linear function, move to a range over the argument of this linear function.
        """
        assert isinstance(element, RangeCondition)
        function = element.function
        if not isinstance(function, Linear):
            return None

        if function.a == 0:
            return TrueCondition if element.range.contains(function.b) else FalseCondition

        return RangeCondition(function.elem, (element.range - function.b) / function.a)

    @staticmethod
    def _binary_combination_linearization(element: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(element, BinaryCombination)
        elem1 = element.c1 * Linear.of(element.elem1)
        elem2 = element.c2 * Linear.of(element.elem2)
        if elem1.elem == elem2.elem:
            return Linear(elem1.a + elem2.a, elem1.elem, elem1.b + elem2.b)
        return None

    # </editor-fold>


RangeCondition.register_class_simplifier(Linear._transform_range)
Linear.register_class_simplifier(Linear._evaluate_simplifier)
Linear.register_class_simplifier(Linear._transform_linear)
BinaryCombination.register_class_simplifier(Linear._binary_combination_linearization)


def _extract_scalar_signed_addition(var_dict: VarDict, sign: int = 1) -> Optional[MapElement]:

    add_vars = get_var_values((Add if sign == 1 else Sub).vars, var_dict)
    if add_vars is None:
        return None

    linear_var1 = Linear.of(add_vars[0])
    linear_var2 = sign * Linear.of(add_vars[1])
    linear_var2 = Linear.of(linear_var2)
    if (linear_var1.a != 0 and linear_var1.b != 0) or (linear_var2.a != 0 and linear_var2.b != 0):
        b = linear_var1.b + linear_var2.b
        # linear_var1 = Linear(linear_var1.a, linear_var1.elem, 0)
        # linear_var2 = Linear(linear_var2.a, linear_var2.elem, 0)
        logger.log(f"Extracted the scalar {green(b)}")
        return ((linear_var1.a * linear_var1.elem) + (linear_var2.a * linear_var2.elem)) + b

    return None


def extract_scalar_addition(var_dict: VarDict) -> Optional[MapElement]:
    return _extract_scalar_signed_addition(var_dict=var_dict, sign=1)


def extract_scalar_subtraction(var_dict: VarDict) -> Optional[MapElement]:
    return _extract_scalar_signed_addition(var_dict=var_dict, sign=-1)


Add.register_simplifier(extract_scalar_addition)
Sub.register_simplifier(extract_scalar_subtraction)
