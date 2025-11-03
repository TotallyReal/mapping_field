import math
from typing import Tuple, Optional, List, Union

from mapping_field.arithmetics import _as_combination
from mapping_field.mapping_field import MapElement, VarDict, MapElementConstant, OutputPromise
from mapping_field.new_conditions import Condition, FalseCondition, TrueCondition, IsCondition

class IntervalRange:

    # TODO: Should I consider this as a MapElement as well? Namely a function which returns 1 on the interval
    #       and 0 otherwise?
    #       Then, the RangeCondition of func becomes the composition interval * func .

    @staticmethod
    def of_point(value: float):
        return IntervalRange(value, value, True, True)

    @staticmethod
    def all():
        return IntervalRange(float('-inf'), float('inf'), False, False)

    @staticmethod
    def empty():
        return IntervalRange(1, 0, False, False)

    def __init__(self, low:float, high: float, contain_low: bool = True, contain_high: bool = False):
        # TODO: make these variables frozen
        self.low = low
        self.high = high
        self.contain_low = contain_low if low != float('-inf') else False
        self.contain_high = contain_high if high != float('inf') else False
        self.is_empty = ((self.high < self.low) or
                         (self.high == self.low and not (self.contain_low and self.contain_high)))
        self.is_point = None
        if self.low == self.high and self.contain_low and self.contain_high:
            self.is_point = self.low

    def __str__(self):
        if self.is_empty:
            return '∅'
        return f'{"[" if self.contain_low else "("}{self.low},{self.high}{"]" if self.contain_high else ")"}'

    def str_middle(self, middle:str):
        if self.is_empty:
            return '!∅!'
        if self.is_point is not None:
            return f'({middle} = {self.is_point})'
        lower = '' if self.low == float('-inf')  else (str(self.low) + ('<=' if self.contain_low else '<'))
        upper = '' if self.high == float('-inf') else (('<=' if self.contain_high else '<') + str(self.high))
        return f'{lower}{middle}{upper}'

    def __eq__(self, other):
        assert isinstance(other, IntervalRange)

        if not self.is_empty and not other.is_empty:
            return (( self.low,  self.high,  self.contain_low,  self.contain_high) ==
                    (other.low, other.high, other.contain_low, other.contain_high))

        return self.is_empty == other.is_empty

    def contains(self, other: Union[int, float, 'IntervalRange']) -> bool:
        if not isinstance(other, IntervalRange):
            other = IntervalRange.of_point(other)

        if self.is_empty:
            return other.is_empty
        if other.is_empty:
            return True

        if other.low < self.low or self.high < other.high:
            return False

        if other.low == self.low and other.contain_low and not self.contain_low:
            return False

        if other.high == self.high and other.contain_high and not self.contain_high:
            return False

        return True

    def complement(self) -> Tuple['IntervalRange', 'IntervalRange']:
        if self.is_empty:
            return (self, IntervalRange.all())
        return (IntervalRange(low=float('-inf'), high=self.low, contain_low=False, contain_high=not self.contain_low) ,
                IntervalRange(low=self.high, high=float('inf'), contain_low=not self.contain_high, contain_high=False))

    def intersection(self, other: 'IntervalRange') -> Optional['IntervalRange']:
        """
        Returns the intersection interval if not empty, otherwise return None.
        """
        range1 = self
        range2 = other

        if range1.is_empty or range2.is_empty:
            return None

        low  = max(range1.low,  range2.low)
        high = min(range1.high, range2.high)
        if high < low:
            return None

        contain_low  = range1.contains(low) and range2.contains(low)
        contain_high = range1.contains(high) and range2.contains(high)

        if high == low:
            if not contain_low:
                return None
            return IntervalRange.of_point(low)

        return IntervalRange(low, high, contain_low, contain_high)

    def union(self, other: 'IntervalRange') -> Optional['IntervalRange']:
        """
        Returns the union interval if can be written as an interval, otherwise return None.
        """

        range1 = self
        range2 = other

        if range1.is_empty:
            return range2
        if range2.is_empty:
            return range1

        low = max(range1.low, range2.low)
        high = min(range1.high, range2.high)
        if high < low:
            return None

        contain_mid = range1.contains(low) or range2.contains(low)

        if high == low and not contain_mid:
            # missing one point
            return None

        true_low = min(range1.low, range2.low)
        contain_true_low = range1.contains(true_low) or range2.contains(true_low)
        true_high = max(range1.high, range2.high)
        contain_true_high = range1.contains(true_high) or range2.contains(true_high)

        return IntervalRange(true_low, true_high, contain_true_low, contain_true_high)

    def __add__(self, value: Union[int, float]) -> 'IntervalRange':
        if value == 0:
            return self
        return IntervalRange(self.low + value, self.high + value, self.contain_low, self.contain_high)

    def __radd__(self, value: Union[int, float]) -> 'IntervalRange':
        return self.__add__(value)

    def __sub__(self, value: Union[int, float]) -> 'IntervalRange':
        return self.__add__(-value)

    def __mul__(self, value: Union[int, float]) -> 'IntervalRange':
        if self.is_empty:
            return self
        if value > 0:
            return IntervalRange(self.low * value, self.high * value, self.contain_low, self.contain_high)
        if value < 0:
            return IntervalRange(self.high * value, self.low * value, self.contain_high, self.contain_low)

        # if value == 0:
        assert self.contain_low and self.contain_high
        return IntervalRange.of_point(0)

    def __rmul__(self, value: Union[int, float]) -> 'IntervalRange':
        return self.__mul__(value)

    def __truediv__(self, value: Union[int, float]) -> 'IntervalRange':
        assert value != 0, 'DO NOT DIVIDE BY ZERO'
        return self.__mul__(1/value)

Range = Tuple[float, float]

class InRange(OutputPromise):

    def __init__(self, f_range: Union[IntervalRange, Tuple[float, float]]):
        super().__init__()
        self.range = f_range if isinstance(f_range, IntervalRange) else IntervalRange(*f_range)

    @staticmethod
    def consolidate_ranges(element:MapElement) -> Tuple[int, IntervalRange]:
        f_range = IntervalRange.all()
        count = 0
        promises = []
        for in_range in element.output_promises(of_type=InRange):
            promises.append(in_range)
            count += 1
            f_range = f_range.intersection(in_range.range)
            if f_range is None:
                raise Exception(f'{element} range promises collapse to an empty range')
        if count > 1:
            element.remove_promises(promises)
            element.add_promise(InRange(f_range))
        return count, f_range


# <editor-fold desc=" --------------- RangeCondition ---------------">

class RangeCondition(Condition):

    def __init__(self, function: MapElement, f_range: Union[IntervalRange, Tuple[float, float]]):
        super().__init__(function.vars)
        self.function = function
        self.range = f_range if isinstance(f_range, IntervalRange) else IntervalRange(*f_range)
        self.add_promise(IsCondition)

    def to_string(self, vars_str_list: List[str]):
        return self.range.str_middle(repr(self.function))

    def __eq__(self, condition: MapElement) -> bool:
        if isinstance(condition, RangeCondition):
            return self.function == condition.function and self.range == condition.range
        return super().__eq__(condition)

    def __hash__(self) -> int:
        # TODO: Maybe have a better hash?
        return id(self)

    def invert(self) -> Optional[MapElement]:
        range1, range2 = self.range.complement()
        upper = RangeCondition(self.function, range1)
        lower = RangeCondition(self.function, range2)
        return upper | lower

    def and_(self, condition: MapElement) -> Optional[MapElement]:
        if isinstance(condition, RangeCondition) and condition.function == self.function:
            f_range = self.range.intersection(condition.range)
            return FalseCondition if (f_range is None) else RangeCondition(condition.function, f_range)

        return None

    def or_(self, condition: MapElement) -> Optional[MapElement]:
        if isinstance(condition, RangeCondition) and condition.function == self.function:
            f_range = self.range.union(condition.range)
            return None if (f_range is None) else RangeCondition(condition.function, f_range)

        return None

    # <editor-fold desc=" ======= Simplifiers ======= ">

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
        if self.range.is_empty:
            return FalseCondition
        if self.range.low == float('-inf') and self.range.high == float('inf'):
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
        return TrueCondition if element.range.contains(value) else FalseCondition

    @staticmethod
    def _ranged_promise_simplifier(element: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(element, RangeCondition)
        num_range_promises, f_range = InRange.consolidate_ranges(element.function)
        if num_range_promises == 0:
            return None

        if element.range.contains(f_range):
            return TrueCondition

        f_range = f_range.intersection(element.range)
        if f_range is None:
            return FalseCondition

        if f_range != element.range:
            return RangeCondition(element.function, f_range)

        if num_range_promises > 1:
            # TODO: Element is "simpler" since its function has less (but equivalent) promises.
            #       I prefer to not change the function, but return a new one. Should think if this is
            #       really needed.
            return element

        return None

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

        if c1 == 0:
            # Should have been caught in the _evaluated_simplifier, but just in case:
            return TrueCondition if element.range.contains(c2) else FalseCondition

        f_range = (element.range - c2) / c1

        return RangeCondition(elem1, f_range)
    # </editor-fold>

RangeCondition.register_class_simplifier(RangeCondition._evaluated_simplifier)
RangeCondition.register_class_simplifier(RangeCondition._ranged_promise_simplifier)
RangeCondition.register_class_simplifier(RangeCondition._linear_combination_simplifier)

def _ranged(elem: MapElement, low: int, high: int, contains_low: bool = True, contains_high: bool = False) -> RangeCondition:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return RangeCondition(elem, IntervalRange(low, high, contains_low, contains_high))
    return NotImplemented

MapElement.__le__ = lambda self, n: _ranged(self, float('-inf'), n, False, True)
MapElement.__lt__ = lambda self, n: _ranged(self, float('-inf'), n, False, False)
MapElement.__ge__ = lambda self, n: _ranged(self, n,  float('inf'), True,  False)
MapElement.__gt__ = lambda self, n: _ranged(self, n,  float('inf'), False, False)


MapElement.__lshift__ = lambda self, n: RangeCondition(self, IntervalRange.of_point(n))

# </editor-fold>