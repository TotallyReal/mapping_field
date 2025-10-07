from abc import abstractmethod
import math
from typing import List, Tuple

from mapping_field import MapElement, VarDict, FuncDict, MapElementConstant, ExtElement
from mapping_field.conditions import Condition, TrueCondition, FalseCondition
from mapping_field.ranged_condition import RangeCondition, RangeTransformer


class LinearTransformer:

    @abstractmethod
    def transform_linear(self, a: int, b: int) -> Tuple[int, MapElement, int]:
        pass


class Linear(MapElement, RangeTransformer):

    @staticmethod
    def of(elem: MapElement):
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
        return Linear(self.a, self.elem._call_with_dict(var_dict, func_dict), self.b)

    def _simplify_with_var_values(self, var_dict: VarDict) -> 'MapElement':
        if self.a == 0:
            return MapElementConstant(self.b)

        elem = self.elem._simplify_with_var_values(var_dict)
        if isinstance(elem, MapElementConstant):
            return MapElementConstant(self.a * elem.evaluate() + self.b)

        if isinstance(elem, LinearTransformer):
            a, elem, b = elem.transform_linear(self.a, self.b)
            if a == 0:
                return MapElementConstant(b)
            return Linear(a, elem, b)

        return Linear(self.a, elem, self.b)

    # <editor-fold desc=" ------------------------ Arithmetics ------------------------">

    def __neg__(self):
        return Linear(-self.a, self.elem, -self.b)

    # The Linear addition always tries to return a "simplified" Linear map.
    def add(self, other: MapElement) -> MapElement:
        try:
            value = other.evaluate() if isinstance(other, MapElement) else other
            if value == 0:
                return self
            return Linear(self.a, self.elem, self.b + value)
        except:
            pass

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

                elem1 *= (a1//gcd)
                elem2 *= (a2//gcd)
                result = Linear(gcd, elem1 + elem2, b1 + b2)
                return result

            # and self.elem == other.elem:

        # result = self._try_add_binary_expansion(other)
        return super().add(other) # if result is None else result

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, MapElementConstant) and isinstance(other.elem, (int, float)):
            other = other.elem
        if isinstance(other, (int, float)):
            if other == 1:
                return self
            if other == 0:
                return MapElementConstant.zero
            return Linear(self.a * other, self.elem, self.b * other)
        return super().__mul__(other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, MapElementConstant) and isinstance(other.elem, (int, float)):
            other = other.elem
        if isinstance(other, (int, float)):
            return Linear(self.a / other, self.elem, self.b / other)
        return super().__truediv__(other)

    # </editor-fold>

    def evaluate(self) -> ExtElement:
        if isinstance(self.elem, MapElementConstant):
            return self.a * self.elem.evaluate() + self.b

        assert self.a == 0
        return self.b

    def __eq__(self, other: MapElement):
        if self.elem == other:
            return self.a == 1 and self.b == 0
        if not isinstance(other, Linear):
            return super().__eq__(other)

        return (self.a == other.a) and (self.b == other.b) and (self.elem.simplify() == other.elem.simplify())

    def transform_range(self, f_range:Tuple[float, float]) -> Condition:
        l, h = f_range
        if self.a == 0:
            return TrueCondition if l<= self.b < h else FalseCondition

        f_range = ((l-self.b)/self.a, (h-self.b)/self.a)
        if self.a < 0:
            f_range = (f_range[1], f_range[0])
        return RangeCondition(self.elem, f_range)
