from typing import Tuple, Optional, List

from mapping_field.mapping_field import MapElement
from mapping_field.new_conditions import Condition, FalseCondition
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


def _ranged(elem: MapElement, low: int, high: int) -> RangeCondition:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return RangeCondition(elem, (low, high))
    return NotImplemented

MapElement.__le__ = lambda self, n: _ranged(self, float('-inf'), n+1)
MapElement.__lt__ = lambda self, n: _ranged(self, float('-inf'), n)
MapElement.__ge__ = lambda self, n: _ranged(self, n, float('inf'))
MapElement.__gt__ = lambda self, n: _ranged(self, n+1, float('inf'))