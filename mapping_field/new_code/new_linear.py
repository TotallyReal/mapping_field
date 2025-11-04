from abc import abstractmethod
import math
from typing import List, Tuple, Optional

from mapping_field.linear import LinearTransformer
from mapping_field.log_utils.tree_loggers import TreeLogger
from mapping_field.arithmetics import _as_combination, Add, Sub
from mapping_field.new_code.new_conditions import FalseCondition, TrueCondition
from mapping_field.new_code.new_ranged_condition import RangeCondition
from mapping_field.serializable import DefaultSerializable
from mapping_field.mapping_field import MapElement, VarDict, FuncDict, MapElementConstant, ExtElement, get_var_values

logger = TreeLogger(__name__)

class BinaryCombination(MapElement):
    # TODO: Right now this is ONLY used for the simplification process inside Linear. Don't generate it for other
    #       reasons.
    #       Later, I should just make a LinearCombination element for expressions of the form sum c_i f_i

    def __init__(self, c1: int, elem1: MapElement, c2: int, elem2: MapElement):
        super().__init__(
            list(set(elem1.vars + elem2.vars))
        )
        self.c1 = c1
        self.elem1 = elem1
        self.c2 = c2
        self.elem2 = elem2

    def to_string(self, vars_str_list: List[str]):
        return f'Comb[{self.c1}*{self.elem1}+{self.c2}*{self.elem2}]'

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
        if self.c1 == 0:
            return Linear(self.c2, self.elem2)
        if self.c2 == 0:
            return Linear(self.c1, self.elem1)

        elem1 = self.elem1._simplify2(var_dict)
        elem2 = self.elem2._simplify2(var_dict)
        if elem1 is not None or elem2 is not None:
            elem1 = elem1 or self.elem1
            elem2 = elem2 or self.elem2
            return BinaryCombination(self.c1, elem1, self.c2, elem2)
        return None

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

    def to_string(self, vars_str_list: List[str]):
        a_str = f'{str(self.a)}*' if self.a != 1 else ''
        b_str = ''
        if self.b > 0:
            b_str = f' + {self.b}'
        if self.b < 0:
            b_str = f' - {-self.b}'
        return f'Lin[{a_str}{self.elem}{b_str}]'

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
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

            bin_comb = BinaryCombination(a1//gcd, elem1, a2//gcd, elem2)
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

    def __eq__(self, other: MapElement):
        if self.elem == other:
            return self.a == 1 and self.b == 0
        if not isinstance(other, Linear):
            return super().__eq__(other)

        return (self-other).evaluate() == 0

    # <editor-fold desc=" ======= Simplifiers ======= ">

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        if self.a == 0:
            return MapElementConstant(self.b)

        elem = self.elem._simplify_with_var_values2(var_dict)
        return None if elem is None else Linear(self.a, elem, self.b)

    def _transform_linear(element: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(element, Linear)
        if isinstance(element.elem, Linear):
            a, b = element.a, element.b
            a_, b_ = element.elem.a, element.elem.b
            return Linear(int(a * a_), element.elem.elem, int(a * b_ + b))
        return None

    @staticmethod
    def _transform_range(element: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(element, RangeCondition)
        function = element.function
        if not isinstance(function, Linear):
            return None

        if function.a == 0:
            return TrueCondition if element.range.contains(function.b) else FalseCondition

        return RangeCondition(function, (element.range-function.b)/function.a)

    # </editor-fold>

RangeCondition.register_class_simplifier(Linear._transform_range)
Linear.register_class_simplifier(Linear._transform_linear)



