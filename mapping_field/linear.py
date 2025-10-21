from abc import abstractmethod
import math
from typing import List, Tuple, Optional

from mapping_field.arithmetics import _as_combination
from mapping_field.serializable import DefaultSerializable
from mapping_field.mapping_field import MapElement, VarDict, FuncDict, MapElementConstant, ExtElement
from mapping_field.conditions import Condition, TrueCondition, FalseCondition
from mapping_field.ranged_condition import RangeCondition, RangeTransformer


class LinearTransformer:

    def linear_combination(self, k1: int, k2: int, elem2: MapElement) -> Optional[Tuple[int, MapElement]]:
        return None

    @abstractmethod
    def transform_linear(self, a: int, b: int) -> Tuple[int, MapElement, int]:
        pass


class Linear(MapElement, RangeTransformer, DefaultSerializable):

    @staticmethod
    def of(elem: MapElement):
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

    def to_string(self, vars_str_list: List[str]):
        a_str = f'{str(self.a)}*' if self.a != 1 else ''
        b_str = ''
        if self.b > 0:
            b_str = f' + {self.b}'
        if self.b < 0:
            b_str = f' - {-self.b}'
        return f'{a_str}{self.elem}{b_str}'

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        elem = self.elem._call_with_dict(var_dict, func_dict)
        if elem is self.elem:
            return self
        return Linear(self.a, self.elem._call_with_dict(var_dict, func_dict), self.b)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        if self.a == 0:
            return MapElementConstant(self.b)

        elem = self.elem._simplify_with_var_values2(var_dict) or self.elem

        n = elem.evaluate()
        if n is not None:
            return MapElementConstant(self.a * n + self.b)

        if isinstance(elem, LinearTransformer):
            a, elem, b = elem.transform_linear(self.a, self.b)
            if a == 0:
                return MapElementConstant(b)
            return Linear(a, elem, b)

        return Linear(self.a, elem, self.b)

    # <editor-fold desc=" ------------------------ Arithmetics ------------------------">

    def neg(self) -> Optional[MapElement]:
        return Linear(-self.a, self.elem, -self.b)

    # The Linear addition always tries to return a "simplified" Linear map.
    def add(self, other: MapElement) -> Optional[MapElement]:
        value = other.evaluate() if isinstance(other, MapElement) else other
        if isinstance(value, int):
            return Linear(self.a, self.elem, self.b + value)

        if other == self.elem:
            return Linear(self.a + 1, self.elem, self.b)

        if isinstance(other, Linear):
            if isinstance(other.elem, LinearTransformer):
                a2, elem2, b2 = other.elem.transform_linear(other.a, other.b)
            else:
                a2, elem2, b2 = other.a, other.elem, other.b

            if isinstance(self.elem, LinearTransformer):
                a1, elem1, b1 = self.elem.transform_linear(self.a, self.b)
            else:
                a1, elem1, b1 = self.a, self.elem, self.b

            if elem1 == elem2:
                return Linear(a1 + a2, elem1, b1 + b2)

            if isinstance(a1, int) and isinstance(a2, int):
                gcd = math.gcd(a1, a2)

                if isinstance(elem1, LinearTransformer):
                    result = elem1.linear_combination(a1//gcd, a2//gcd, elem2)
                    if result is not None:
                        k, elem = result
                        return Linear(gcd * k, elem, b1 + b2)

                if isinstance(elem2, LinearTransformer):
                    result = elem2.linear_combination(a2//gcd, a1//gcd, elem1)
                    if result is not None:
                        k, elem = result
                        return Linear(gcd * k, elem, b1 + b2)

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

    def __eq__(self, other: MapElement):
        if self.elem == other:
            return self.a == 1 and self.b == 0
        if not isinstance(other, Linear):
            return super().__eq__(other)

        return (self-other).evaluate() == 0


    def transform_range(self, f_range:Tuple[float, float]) -> Condition:
        l, h = f_range
        if self.a == 0:
            return TrueCondition if l<= self.b < h else FalseCondition

        f_range = ((l-self.b)/self.a, (h-self.b)/self.a)
        if self.a < 0:
            f_range = (f_range[1]+1, f_range[0]+1)
        return RangeCondition(self.elem, f_range)
