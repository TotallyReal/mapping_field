from typing import Tuple, Optional, List

from mapping_field.mapping_field import MapElement
from mapping_field.new_conditions import Condition, FalseCondition

Range = Tuple[float, float]

class RangeCondition(Condition):

    def __init__(self, function: MapElement, f_range: Range, simplified: bool = False):
        super().__init__(function.vars)
        self.function = function
        self.range = f_range

    def to_string(self, vars_str_list: List[str]):
        lower = '' if self.range[0] == float('-inf') else f'{self.range[0]} <= '
        upper = '' if self.range[1] == float('inf') else f' < {self.range[1]}'
        return f'{lower}{repr(self.function)}{upper}'

    def __eq__(self, condition: MapElement) -> bool:
        if isinstance(condition, RangeCondition):
            return self.function == condition.function and self.range == condition.range
        return super().__eq__(condition)

    def and_(self, condition: MapElement) -> Optional[MapElement]:
        if isinstance(condition, RangeCondition) and condition.function == self.function:
            low = max(self.range[0], condition.range[0])
            high = min(self.range[1], condition.range[1])
            if high <= low:
                return FalseCondition
            return RangeCondition(self.function, (low, high))

        return None