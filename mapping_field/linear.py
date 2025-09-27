import math
from typing import List, Optional, Dict, Tuple

from mapping_field import MapElement, Var, VarDict, FuncDict, MapElementConstant
from mapping_field.conditions import (
    RangeCondition, RangeTransformer, AssignmentCondition, Condition, TrueCondition, FalseCondition)


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
    # def __str__(self):
        a_str = f'{str(self.a)}*' if self.a != 1 else ''
        b_str = ''
        if self.b > 0:
            b_str = f' + {self.b}'
        if self.b < 0:
            b_str = f' - {-self.b}'
        return f'{a_str}{self.elem}{b_str}'

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        return Linear(self.a, self.var._call_with_dict(var_dict, func_dict), self.b)

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        if self.a == 0:
            return MapElementConstant(self.b)

        elem = self.elem._simplify_with_entries(simplified_entries)
        return Linear(self.a, elem, self.b)

    # <editor-fold desc=" ------------------------ Arithmetics ------------------------">

    def __neg__(self):
        return Linear(-self.a, self.elem, -self.b)

    def __add__(self, other):
        if isinstance(other, MapElementConstant) and isinstance(other, (int, float)):
            other = other.elem
        if isinstance(other, (int, float)):
            return Linear(self.a, self.elem, self.b + other)
        if other == self.elem:
            return Linear(self.a + 1, self.elem, self.b)
        if isinstance(other, Linear) and self.elem == other.elem:
            return Linear(self.a + other.a, self.elem, self.b + other.b)
        return super().__add__(other)

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

    def __eq__(self, other: MapElement):
        if not isinstance(other, Linear):
            return super().__eq__(other)

        return (self.a == other.a) and (self.b == other.b) and (self.elem == other.elem)

    def transform_range(self, f_range:Tuple[float, float]) -> Condition:
        l, h = f_range
        if self.a == 0:
            return TrueCondition if l<= self.b < h else FalseCondition

        f_range = ((l-self.b)/self.a, (h-self.b)/self.a)
        if self.a < 0:
            f_range = (f_range[1], f_range[0])
        if isinstance(self.elem, Linear):
            return self.elem.transform_range(f_range)
        else:
            return RangeCondition(self.elem, f_range)


class BoolVar(Var):

    def __new__(cls, var_name: str):
        return super(BoolVar, cls).__new__(cls, var_name)

    def __init__(self, var_name: str):
        super().__init__(var_name)


class IntVar(Var, RangeTransformer):

    def __new__(cls, var_name: str):
        return super(IntVar, cls).__new__(cls, var_name)

    def __init__(self, var_name: str):
        super().__init__(var_name)

    def transform_range(self, f_range:Tuple[float, float]) -> RangeCondition:

        l, h = f_range
        k = math.ceil(l)
        if k < h <= k+1:
            return AssignmentCondition({self: k})

        return RangeCondition(self, f_range)

# '''
# A ranged function f(x) with range I is defined as f(x) if this value is in I and 0 otherwise, namely:
#
#         f(x) * 1( f(x) in I )
# '''
# class RangedFunction:
#
#     def __init__(self, function: MapElement, f_range: Range):
#         self.function = function
#         self.range = f_range
#         self.indicators = []
#
# class Number2(Var):
#
#     def __new__(cls, var_name: str, coefs = None):
#         return super(Number2, cls).__new__(cls, var_name)
#
#     def __init__(self, var_name: str, coefs = None):
#         super().__init__(var_name)
#         if coefs is None:
#             self.coef: List[int] = [0]*5
#             self._range = (0,1)
#         else:
#             assert all((isinstance(c,int) and c>=0) for c in coefs)
#             self.coef: List[int] = coefs
#             max_value = 0
#             power = 1
#             for c in coefs:
#                 max_value += min(c,1)*power
#                 power *= 2
#                 self._range = (0, max_value)
#
#     def intersect_range(self, l: int, h: int) -> 'Number2':
#         new_l = max(l, self._range[0])
#         new_h = min(h, self._range[1])
#         assert new_l < new_h, f'problem with the range ({new_l},{new_h})'
#         self._range = (new_l, new_h)
#         coefs = self.coef.copy()
#         power = 2 ** len(coefs)
#         for i in range(len(coefs)-1, -1, -1):
#             power //= 2
#             if power >= new_h:
#                 assert coefs[i] != 1
#                 coefs[i] = 0
#                 continue
#             break # TODO: continue this loop
#         return Number2(self.name, coefs)
#
#     def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
#         value = var_dict.get(self)
#         if value is None:
#             return self
#
#         value: Number2 = value
#         range = value.range()
#         if range == self.range():
#             return self
#
#     def range(self):
#         return self._range
#
#     def to_linear(self):
#         return LinearNumber2(1, self, 0)
#
#     @staticmethod
#     def _convert(f):
#         def wrapper(self, other):
#             if isinstance(other, Number2):
#                 return f(self, other.to_linear())
#             return f(self, other)
#         return wrapper
#
#     def __neg__(self):
#         return -self.to_linear()
#
#     def __add__(self, other):
#         return self.to_linear() + other
#
#     def __radd__(self, other):
#         return self.to_linear() + other
#
#     def __sub__(self, other):
#         return self.to_linear() - other
#
#     def __rsub__(self, other):
#         return self - other.to_linear()
#
#     def __mul__(self, other):
#         return self.to_linear() * other
#
#     def __rmul__(self, other):
#         return self.to_linear() * other
#
# class LinearNumber2(MapElement):
#
#     def __init__(self, a: int, num2: Number2, b: int):
#         super().__init__([num2])
#         self.a = a
#         self.b = b
#         self.var = num2
#
#     def range(self):
#         # TODO: When a>1, the upper bound is smaller, since we are using integers
#         if self.a > 0:
#             l, h = self.var.range()
#             return (self.a*l+self.b, self.a*h+self.b)
#
#         if self.a == 0:
#             return (self.b, self.b+1)
#
#         if self.a < 0:
#             l, h = self.var.range()
#             return (self.a*h+self.b, self.a*l+self.b)
#
#     def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
#         return LinearNumber2(self.a, self.var._call_with_dict(var_dict, func_dict), self.b)
#
#     def to_string(self, vars_str_list: List[str]):
#     # def __str__(self):
#         a_str = str(self.a) if self.a != 1 else ''
#         b_str = ''
#         if self.b > 0:
#             b_str = f' + {self.b}'
#         if self.b < 0:
#             b_str = f' - {-self.b}'
#         return f'{a_str}{self.var.to_string(vars_str_list)}{b_str}'
#
#     @Number2._convert
#     def __neg__(self):
#         return LinearNumber2(-self.a, self.var, -self.b)
#
#     @Number2._convert
#     def __add__(self, other):
#         if isinstance(other, int):
#             return LinearNumber2(self.a, self.var, self.b + other)
#         if isinstance(other, LinearNumber2) and self.var == other.var:
#             # TODO: improve equality between variables
#             return LinearNumber2(self.a + other.a, self.var, self.b + other.b)
#         return NotImplemented
#
#     @Number2._convert
#     def __radd__(self, other):
#         return self + other
#
#     @Number2._convert
#     def __sub__(self, other):
#         if isinstance(other, int):
#             return LinearNumber2(self.a, self.var, self.b - other)
#         if isinstance(other, LinearNumber2) and self.var == other.var:
#             # TODO: improve equality between variables
#             return LinearNumber2(self.a - other.a, self.var, self.b - other.b)
#         return NotImplemented
#
#     @Number2._convert
#     def __rsub__(self, other):
#         return other + (-self)
#
#     def __mul__(self, other):
#         if isinstance(other, int):
#             return LinearNumber2(self.a * other, self.var, self.b * other)
#         return NotImplemented
#
#     def __rmul__(self, other):
#         return self * other
#
#
# class _ReLU(MapElementFromFunction):
#
#     # TODO: consider transform constant(-1) into -constant(1)
#     def __init__(self):
#         super().__init__('ReLU', lambda a: a if a>=0 else 0)
#
#     def to_string(self, entries: List[str]):
#         return f'ReLU({entries[0]})'
#
#     def _simplify_partial_constant(self, entries: List[MapElement]) -> MapElement:
#         if isinstance(entries[0], Ranged):
#             range = entries[0].range()
#             if range[0] >= 0:
#                 return entries[0]
#             if range[1] <= 0:
#                 return MapElementConstant(0)
#         return super()._simplify_partial_constant(entries)
#
# ReLU = _ReLU()
#
