import math
from typing import Tuple, Optional, List

from mapping_field.arithmetics import _as_combination
from mapping_field.mapping_field import MapElement, VarDict, MapElementConstant
from mapping_field.new_conditions import Condition, FalseCondition, TrueCondition, IsCondition

Range = Tuple[float, float]

class RangeCondition(Condition):

    def __init__(self, function: MapElement, f_range: Range, simplified: bool = False):
        super().__init__(function.vars)
        self.function = function
        self.range = f_range
        self.add_promise(IsCondition)

    def to_string(self, vars_str_list: List[str]):
        lower = '' if self.range[0] == float('-inf') else f'{self.range[0]} <= '
        upper = '' if self.range[1] == float('inf') else f' < {self.range[1]}'
        return f'{lower}{repr(self.function)}{upper}'

    def __eq__(self, condition: MapElement) -> bool:
        if isinstance(condition, RangeCondition):
            return self.function == condition.function and self.range == condition.range
        return super().__eq__(condition)

    def invert(self) -> Optional[MapElement]:
        low, high = self.range
        upper = RangeCondition(self.function, (high, float('inf')))
        lower = RangeCondition(self.function, (float('-inf'), low))
        return upper | lower

    def and_(self, condition: MapElement) -> Optional[MapElement]:
        if isinstance(condition, RangeCondition) and condition.function == self.function:
            low = max(self.range[0], condition.range[0])
            high = min(self.range[1], condition.range[1])
            if high <= low:
                return FalseCondition
            return RangeCondition(self.function, (low, high))

        return None

    def or_(self, condition: MapElement) -> Optional[MapElement]:
        if isinstance(condition, RangeCondition) and condition.function == self.function:
            if self.range[1] < condition.range[0] or condition.range[1] < self.range[0]:
                return None
            a = min(self.range[0], condition.range[0])
            b = max(self.range[1], condition.range[1])
            return RangeCondition(self.function, (a, b))

        return None

    # <editor-fold desc="Simplifiers">

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
        if self.range[1] <= self.range[0]:
            return FalseCondition
        if self.range[0] == float('-inf') and self.range[1] == float('inf'):
            return TrueCondition

        simplified_function = self.function._simplify2(var_dict)
        if simplified_function is not None:
            return RangeCondition(simplified_function, self.range)
        return None

    @staticmethod
    def _evaluated_simplifier(element: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(element, RangeCondition)
        value = element.function.evaluate()
        if value is None:
            return None
        return TrueCondition if element.range[0] <= value < element.range[1] else FalseCondition

    @staticmethod
    def _linear_combination_simplifier(element: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(element, RangeCondition)
        c1, elem1, c2, elem2 = _as_combination(element.function)

        if c1 == 1 and c2 == 0:
            # Trivial combination
            return None

        if elem2 is not MapElementConstant.one:
            # Too complicated combination
            return None

        # combination is c1*elem + c2.

        low, high = element.range

        if c1 == 0:
            # Should have been caught in the _evaluated_simplifier, but just in case:
            return TrueCondition if low <= c2 < high else FalseCondition

        # TODO: We assume that all the functions are integral for now. Use a Promise to implement it
        low = (low-c2)/c1
        high = ((high-1)-c2)/c1
        if c1 < 0:
            low, high = high, low
        # both ends included
        low = int(math.floor(low))
        if high == int(high):
            high = 1 + int(high)
        else:
            high = int(math.ceil(high))

        return RangeCondition(elem1, (low, high))
    # </editor-fold>

RangeCondition.register_class_simplifier(RangeCondition._evaluated_simplifier)
RangeCondition.register_class_simplifier(RangeCondition._linear_combination_simplifier)


def _ranged(elem: MapElement, low: int, high: int) -> RangeCondition:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return RangeCondition(elem, (low, high))
    return NotImplemented

MapElement.__le__ = lambda self, n: _ranged(self, float('-inf'), n+1)
MapElement.__lt__ = lambda self, n: _ranged(self, float('-inf'), n)
MapElement.__ge__ = lambda self, n: _ranged(self, n, float('inf'))
MapElement.__gt__ = lambda self, n: _ranged(self, n+1, float('inf'))