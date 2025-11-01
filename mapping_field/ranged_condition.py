from abc import abstractmethod
from typing import Tuple, Optional, List, Callable

from mapping_field import CompositionFunction
from mapping_field.mapping_field import VarDict, MapElement, Var
from mapping_field.conditions import Condition, FalseCondition, ConditionIntersection, \
    MapElementProcessor, TrueCondition, NotCondition
from mapping_field.serializable import DefaultSerializable


class _AssignmentCondition(Condition, MapElementProcessor):
    # TODO: to delete?

    def __init__(self, var_dict: VarDict):
        super().__init__(list(var_dict.keys()))
        self.var_dict = var_dict

    def __repr__(self):
        return repr(self.var_dict)

    def _eq_simplified(self, condition: 'Condition') -> bool:
        if isinstance(condition, _AssignmentCondition):
            return self.var_dict == condition.var_dict
        return super()._eq_simplified(condition)

    def process_function(self, func: MapElement) -> MapElement:
        return func(self.var_dict)

    def and_simpler(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, _AssignmentCondition):
            # Combine the dictionaries, if there are no different assignments for the same variables
            for key, value in self.var_dict.items():
                if condition.var_dict.get(key, value) != value:
                    return FalseCondition, True
            return _AssignmentCondition({**self.var_dict, **condition.var_dict}), True

        return super().and_simpler(condition)

    def or_simpler(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, RangeCondition):
            return condition.or_simpler(self)

        cond1 = self
        cond2 = condition

        if isinstance(condition, _AssignmentCondition):
            assignments1 = self.var_dict
            assignments2 = condition.var_dict

            if len(assignments2) < len(assignments1):
                assignments1, assignments2 = assignments2, assignments1
                cond1, cond2 = cond2, cond1

            # check if one condition is contained in the second
            if len(assignments1) < len(assignments2):
                for key, value in assignments1.items():
                    if (key not in assignments2) or assignments1[key] != assignments2[key]:
                        return super().or_simpler(condition)
                return cond1, True

            # same key, only one differ in value, and they are consecutive integers
            if len(assignments1) == len(assignments2):
                special_key = None
                range = (0,0)
                for key, value in assignments1.items():
                    if key not in assignments2:
                        return super().or_simpler(condition)

                    value2 = assignments2[key]
                    if value2 == value:
                        continue
                    if special_key is not None:
                        return super().or_simpler(condition)

                    if abs(value - value2) != 1:
                        return super().or_simpler(condition)
                    special_key = key
                    range = (min(value, value2), max(value, value2) + 1)

                if special_key is None:
                    return self, True

                assign_cond = _AssignmentCondition({k: v for k,v in assignments1.items() if k != special_key})
                range_cond = RangeCondition(special_key, range)
                return ConditionIntersection([assign_cond, range_cond]), True

        return super().or_simpler(condition)

    def simplify(self) -> 'Condition':
        if len(self.var_dict) == 0:
            return TrueCondition
        return self


class SingleAssignmentCondition(Condition, MapElementProcessor, DefaultSerializable):

    @staticmethod
    def from_assignment_dict(var_dict: VarDict) -> Condition:
        condition = TrueCondition
        for v, value in var_dict.items():
            condition &= SingleAssignmentCondition(v, value)
        return condition

    @staticmethod
    def as_assignment_dict(condition: Condition) -> Optional[VarDict]:
        """
        Tries to convert the condition into an assignment, and if so returns it, otherwise returns None.
        """
        if isinstance(condition, SingleAssignmentCondition):
            return condition.var_dict
        if (isinstance(condition, ConditionIntersection) and
            all([isinstance(cond, SingleAssignmentCondition) for cond in condition.conditions])):
            return {cond.var: cond.value for cond in condition.conditions}
        return None

    def __init__(self, v: Var, value: int):
        super().__init__([v])
        self.var = v
        self.value = value
        self.var_dict = {v: value}

    @classmethod
    def serialization_name_conversion(self):
        return {'v' : 'var'}

    def __repr__(self):
        return repr(self.var_dict)

    def _eq_simplified(self, condition: 'Condition') -> bool:
        if isinstance(condition, SingleAssignmentCondition) or isinstance(condition, _AssignmentCondition):
            return self.var_dict == condition.var_dict
        return super()._eq_simplified(condition)

    def process_function(self, func: MapElement) -> MapElement:
        return func(self.var_dict)

    def and_simpler(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, SingleAssignmentCondition) and self.var == condition.var:
            return (self if (self.value == condition.value) else FalseCondition), True

        return super().and_simpler(condition)

    def or_simpler(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, RangeCondition):
            return condition.or_simpler(self)

        if isinstance(condition, SingleAssignmentCondition) and self.var == condition.var:
            a = min(self.value, condition.value)
            b = max(self.value, condition.value)
            if a == b:
                return self, True
            if a + 1 == b:
                return RangeCondition(self.var, (a,b+1)), True

        return super().or_simpler(condition)

    def simplify(self) -> 'Condition':
        if len(self.var_dict) == 0:
            return TrueCondition
        return self

Range = Tuple[float, float]

class AsRange:

    # TODO: find a better way to arrange this whole process

    @staticmethod
    def dict_as_range(function: MapElement, var_dict: VarDict) -> Optional[Range]:
        if not all([isinstance(value, int) for value in var_dict.values()]):
            return None
        if isinstance(function, Var) and len(var_dict) == 1 and (function in var_dict):
            value: int = var_dict[function]
            return (value, value+1)

        return None

    @abstractmethod
    def as_range(self, function: MapElement) -> Optional[Range]:
        """
        Tries to convert this object (should be a condition) to a RangedCondition over the given function.
        If possible, returns that condition, otherwise returns None.
        """
        pass

class ConditionToRangeTransformer:

    @abstractmethod
    def as_range(self, condition: Condition) -> Optional[Range]:
        """
        Assuming self is a MapElement, tries to convert the given condition into a range condition.
        If possible, returns that condition, otherwise returns None.
        """

RangeConditionSimplifier = Callable[[MapElement, Range], Optional[Condition]]

class RangeCondition(Condition, AsRange, DefaultSerializable):

    _simplifiers: List[RangeConditionSimplifier] = []

    @classmethod
    def register_simplifier(cls, simplifier: RangeConditionSimplifier):
        cls._simplifiers.append(simplifier)

    def __init__(self, function: MapElement, f_range: Range, simplified: bool = False):
        super().__init__(function.vars)
        self.function = function
        self.range = f_range
        self._simplified = simplified

    @classmethod
    def serialization_name_conversion(self):
        return {'f_range' : 'range', 'simplified' : '_simplified'}

    def __repr__(self):
        lower = '' if self.range[0] == float('-inf') else f'{self.range[0]} <= '
        upper = '' if self.range[1] == float('inf') else f' < {self.range[1]}'
        return f'{lower}{repr(self.function)}{upper}'

        return f'{self.range[0]} <= {repr(self.function)} < {self.range[1]}'

    def _eq_simplified(self, condition: 'Condition') -> bool:
        if isinstance(condition, RangeCondition):
            return self.function == condition.function and self.range == condition.range
        return super()._eq_simplified(condition)

    def as_range(self, function: MapElement) -> Optional[Range]:
        return self.range if self.function == function else None

    def and_simpler(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, RangeCondition):
            if self.function == condition.function:
                low = max(self.range[0], condition.range[0])
                high = min(self.range[1], condition.range[1])
                if high <= low:
                    return FalseCondition, True
                return RangeCondition(self.function, (low, high)), True

        if isinstance(condition, SingleAssignmentCondition):
            function = self.function(condition.var_dict)
            if function != self.function:
                # TODO: beware of infinite loops...
                return ConditionIntersection([RangeCondition(function, self.range), condition]), True

        return super().and_simpler(condition)

    def or_simpler(self, condition: Condition) -> Tuple[Condition, bool]:

        # TODO: For now I assume that we only deal with integers
        if isinstance(condition, SingleAssignmentCondition):
            if condition.var == self.function:
                if condition.value == self.range[0] - 1:
                    return RangeCondition(self.function, (self.range[0]-1, self.range[1])), True
                if condition.value == self.range[1]:
                    return RangeCondition(self.function, (self.range[0], self.range[1]+1)), True
                if self.range[0] <= condition.value < self.range[1]:
                    return self, True

        f_range = None
        if isinstance(condition, AsRange):
            f_range = condition.as_range(self.function)
        elif isinstance(self.function, ConditionToRangeTransformer):
            f_range = self.function.as_range(condition)

        if f_range is not None:
            if self.range[1] < f_range[0] or f_range[1] < self.range[0]:
                return super().or_simpler(condition)
            a = min(self.range[0], f_range[0])
            b = max(self.range[1], f_range[1])
            return RangeCondition(self.function, (a, b)), True

        return super().or_simpler(condition)


    def simplify(self) -> 'Condition':
        condition = self
        while True:

            for simplifier in self.__class__._simplifiers:
                new_condition = simplifier(condition.function, condition.range)
                if new_condition is None:
                    continue
                condition = new_condition
                break
            else:
                return condition

            if (not isinstance(condition, RangeCondition)) or condition._simplified:
                break

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

# TODO: Use the simplifier mechanism from MapElement instead.
original_not_simplifier = NotCondition.simplify
def not_simplifier(self: NotCondition) -> Condition:
    if isinstance(self.condition, RangeCondition):
        a, b = self.condition.range
        upper = RangeCondition(self.condition.function, (b, float('inf'))).simplify()
        lower = RangeCondition(self.condition.function, (float('-inf'), a)).simplify()
        return (upper | lower).simplify()
    return original_not_simplifier(self.condition)
NotCondition.simplify = not_simplifier


def _range_transformer_simplifier(function: MapElement, f_range: Range) -> Optional[Condition]:
    if isinstance(function, RangeTransformer):
        return function.transform_range(f_range)
    return None

def _range_evaluator_simplifier(function: MapElement, f_range: Range) -> Optional[Condition]:
    if f_range[1] <= f_range[0]:
        return FalseCondition
    if f_range[0] == float('-inf') and f_range[1] == float('inf'):
        return TrueCondition
    n = function.evaluate()
    if n is not None:
        return (TrueCondition if f_range[0] <= n < f_range[1] else FalseCondition)
    return None

def _range_additive_simplifier(function: MapElement, f_range: Range) -> Optional[Condition]:
    if not isinstance(function, CompositionFunction):
        return None

    f = function.function
    if f not in (MapElement.addition, MapElement.subtraction, MapElement.multiplication, MapElement.division):
        return None

    entries = function.entries
    n = entries[0].evaluate()
    if n is not None:
        if f in (MapElement.addition, MapElement.subtraction):
            return RangeCondition(entries[1],(f_range[0] - n, f_range[1] - n))

    n = entries[1].evaluate()
    if n is not None:
        if f is MapElement.addition:
            return RangeCondition(entries[0],(f_range[0] - n, f_range[1] - n))
        if f is MapElement.subtraction:
            return RangeCondition(entries[0],(f_range[0] + n, f_range[1] + n))

    return None

RangeCondition.register_simplifier(_range_transformer_simplifier)
RangeCondition.register_simplifier(_range_evaluator_simplifier)
RangeCondition.register_simplifier(_range_additive_simplifier)


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

Var.__lshift__ = lambda self, n: SingleAssignmentCondition(self, n)

