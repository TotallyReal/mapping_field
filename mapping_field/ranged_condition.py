from abc import abstractmethod
from typing import Tuple, Optional

from mapping_field import VarDict, MapElement, MapElementConstant
from mapping_field.conditions import Condition, FalseCondition, ConditionIntersection, Range, ConditionalFunction, \
    MapElementProcessor, TrueCondition


class AssignmentCondition(Condition, MapElementProcessor):

    def __init__(self, var_dict: VarDict):
        super().__init__(list(var_dict.keys()))
        self.var_dict = var_dict

    def __repr__(self):
        return repr(self.var_dict)

    def _eq_simplified(self, condition: 'Condition') -> bool:
        if isinstance(condition, AssignmentCondition):
            return self.var_dict == condition.var_dict
        return super()._eq_simplified(condition)

    def process(self, func: MapElement) -> MapElement:
        return func(self.var_dict)

    def __and__(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, AssignmentCondition):
            # Combine the dictionaries, if there are no different assignments for the same variables
            for key, value in self.var_dict.items():
                if condition.var_dict.get(key, value) != value:
                    return FalseCondition, True
            return AssignmentCondition({**self.var_dict, **condition.var_dict}), True

        return super().__and__(condition)

    def __or__(self, condition: 'Condition') -> Tuple['Condition', bool]:
        if isinstance(condition, RangeCondition):
            return condition | self

        cond1 = self
        cond2 = condition

        if isinstance(condition, AssignmentCondition):
            assignments1 = self.var_dict
            assignments2 = condition.var_dict

            if len(assignments2) < len(assignments1):
                assignments1, assignments2 = assignments2, assignments1
                cond1, cond2 = cond2, cond1

            # check if one condition is contained in the second
            if len(assignments1) < len(assignments2):
                for key, value in assignments1.items():
                    if (key not in assignments2) or assignments1[key] != assignments2[key]:
                        return super().__or__(condition)
                return cond1, True

            # same key, only one differ in value, and they are consecutive integers
            if len(assignments1) == len(assignments2):
                special_key = None
                range = (0,0)
                for key, value in assignments1.items():
                    if key not in assignments2:
                        return super().__or__(condition)

                    value2 = assignments2[key]
                    if value2 == value:
                        continue
                    if special_key is not None:
                        return super().__or__(condition)

                    if abs(value - value2) != 1:
                        return super().__or__(condition)
                    special_key = key
                    range = (min(value, value2), max(value, value2) + 1)

                if special_key is None:
                    return self, True

                assign_cond = AssignmentCondition({k: v for k,v in assignments1.items() if k != special_key})
                range_cond = RangeCondition(special_key, range)
                return ConditionIntersection([assign_cond, range_cond]), True

        return super().__or__(condition)

    def simplify(self) -> 'Condition':
        if len(self.var_dict) == 0:
            return TrueCondition
        return self


class RangeCondition(Condition):

    def __init__(self, function: MapElement, f_range: Range, simplified: bool = False):
        super().__init__(function.vars)
        self.function = function
        self.range = f_range
        self._simplified = simplified

    def __repr__(self):
        lower = '' if self.range[0] == float('-inf') else f'{self.range[0]} <= '
        upper = '' if self.range[1] == float('inf') else f' < {self.range[1]}'
        return f'{lower}{repr(self.function)}{upper}'

        return f'{self.range[0]} <= {repr(self.function)} < {self.range[1]}'

    def _eq_simplified(self, condition: 'Condition') -> bool:
        if isinstance(condition, RangeCondition):
            return self.function == condition.function and self.range == condition.range
        return super()._eq_simplified(condition)

    def __and__(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, RangeCondition):
            if self.function == condition.function:
                low = max(self.range[0], condition.range[0])
                high = min(self.range[1], condition.range[1])
                if high <= low:
                    return FalseCondition, True
                return RangeCondition(self.function, (low, high)), True

        if isinstance(condition, AssignmentCondition):
            function = self.function(condition.var_dict)
            if function != self.function:
                # TODO: beware of infinite loops...
                return ConditionIntersection([RangeCondition(function, self.range), condition]), True

        return super().__and__(condition)

    def __or__(self, condition: Condition) -> Tuple[Condition, bool]:

        # TODO: For now I assume that we only deal with integers
        if isinstance(condition, AssignmentCondition) and len(condition.var_dict) == 1:
            key, value = list(condition.var_dict.items())[0]
            if key == self.function:
                if value == self.range[0] - 1:
                    return RangeCondition(self.function, (self.range[0]-1, self.range[1])), True
                if value == self.range[1]:
                    return RangeCondition(self.function, (self.range[0], self.range[1]+1)), True
                if self.range[0] <= value < self.range[1]:
                    return self, True


        if isinstance(condition, RangeCondition) and condition.function == self.function:
            if self.range[1] < condition.range[0] or condition.range[1] < self.range[0]:
                return super().__or__(condition)
            a = min(self.range[0], condition.range[0])
            b = max(self.range[1], condition.range[1])
            return RangeCondition(self.function, (a, b)), True

        return super().__or__(condition)


    def simplify(self) -> 'Condition':
        if self.range[0] >= self.range[1]:
            return FalseCondition

        try:
            n = self.function.evaluate()
            return (TrueCondition if self.range[0] <= n < self.range[1] else FalseCondition)
        except:
            pass

        condition = self
        while (isinstance(condition, RangeCondition) and (not condition._simplified) and
               isinstance(condition.function, RangeTransformer)):
            new_condition = condition.function.transform_range(condition.range)
            if new_condition is None:
                return condition
            condition = new_condition

        return condition

    # <editor-fold desc=" ------------------------ comparison ------------------------">

    # Used for writing ranged conditions as 'a < func < b' for some map element func, and two constants a, b

    def __lt__(self, other) -> Condition:
        if not isinstance(other, (int, float)):
            return NotImplemented
        if other == float('inf'):
            return self
        if self.range[1] != float('inf'):
            raise Exception(f'Cannot add a second upper bound {other} to {self}')
        return RangeCondition(self.function, (self.range[0], other))

    def __le__(self, other) -> Condition:
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.__lt__(other + 1)

    def __ge__(self, other) -> Condition:
        if not isinstance(other, (int, float)):
            return NotImplemented
        if other == float('-inf'):
            return self
        if self.range[0] != float('-inf'):
            raise Exception(f'Cannot add a second lower bound {other} to {self}')
        return RangeCondition(self.function, (other, self.range[1]))

    def __gt__(self, other) -> Condition:
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self.__ge__(other - 1)

    # </editor-fold>


def _ranged(elem: MapElement, low: int, high: int) -> RangeCondition:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return RangeCondition(elem, (low, high))
    return NotImplemented

MapElement.__le__ = lambda self, n: _ranged(self, float('-inf'), n+1)
MapElement.__lt__ = lambda self, n: _ranged(self, float('-inf'), n)
MapElement.__ge__ = lambda self, n: _ranged(self, n, float('inf'))
MapElement.__gt__ = lambda self, n: _ranged(self, n+1, float('inf'))


class RangeTransformer:

    @abstractmethod
    def transform_range(self, range_values: Range) -> Optional[Condition]:
        """
        Try to simplify being in the given range (should be called on a MapElement).
        If cannot be simplified, return None.
        """
        pass


def ReLU(map_elem: MapElement):
    zero = MapElementConstant.zero
    if isinstance(map_elem, ConditionalFunction):
        regions = []
        for condition, func in map_elem.regions:
            regions.append( (condition * (func >= 0), func) )
            regions.append( (condition * (func < 0), zero) )
        regions = [(cond, func) for cond, func in regions if FalseCondition != cond]
        return ConditionalFunction(regions)
    return ConditionalFunction([
        ((map_elem >= 0), map_elem),
        ((map_elem < 0), zero)
    ])