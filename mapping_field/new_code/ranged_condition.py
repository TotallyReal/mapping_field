import math
import operator

from typing import Dict, Optional, Tuple, Union

from mapping_field.new_code.arithmetics import Add, Mult, Sub, _as_combination
from mapping_field.new_code.conditions import Condition, FalseCondition, TrueCondition
from mapping_field.new_code.mapping_field import (
    CompositionFunction, FuncDict, MapElement, MapElementConstant, MapElementProcessor,
    OutputPromises, OutputValidator, Var, VarDict, always_validate_promises,
)
from mapping_field.new_code.promises import IsCondition, IsIntegral


class IntervalRange:

    # TODO: Should I consider this as a MapElement as well? Namely a function which returns 1 on the interval
    #       and 0 otherwise?
    #       Then, the RangeCondition of func becomes the composition interval * func and similarly the InRange
    #       Promise only see the composition with interval.
    #       More generally, a function h(x) of one variable can be viewed as a promise on the output of a function
    #       f(x_1,...,x_n) by saying that h(f(x_1,...,x_n)) = 1.

    @staticmethod
    def of_point(value: float):
        return IntervalRange[value, value]

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
        self.is_all = (low == float('-inf') and high == float('inf'))
        self.is_point: Optional[float] = None
        if self.low == self.high and self.contain_low and self.contain_high:
            self.is_point = self.low

    def __class_getitem__(cls, params) -> 'IntervalRange':
        # Normalize to tuple
        if not isinstance(params, tuple):
            params = (params,)

        # Check count
        if len(params) != 2:
            raise TypeError(f"{cls.__name__} expects exactly 2 numeric parameters (got {len(params)})")

        # Check types
        if not all(isinstance(p, (int, float)) for p in params):
            raise TypeError(f"{cls.__name__} expects int or float parameters (got {params!r})")

        return IntervalRange(params[0], params[1], True, True)

    def __str__(self):
        if self.is_empty:
            return '∅'
        return f'{"[" if self.contain_low else "("}{self.low},{self.high}{"]" if self.contain_high else ")"}'

    def str_middle(self, middle:str):
        if self.is_empty:
            return '!∅!'
        if self.is_point is not None:
            return f'({middle} = {self.is_point})'
        # TODO: I can use ≤ instead of <=. However, it doesn't always looks good in terminal, and sometimes I
        #       just see the ASCII \u2264 instead.
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

    def integral_union(self, other: 'IntervalRange') -> Optional['IntervalRange']:
        range1 = self.as_integral(half_open=True)
        range2 = other.as_integral(half_open=True)
        result = range1.union(range2)
        return None if result is None else result.as_integral(half_open=False)

    def __neg__(self):
        return IntervalRange(-self.high, -self.low, self.contain_high, self.contain_low)

    def __add__(self, value: Union[int, float, 'IntervalRange']) -> 'IntervalRange':
        if not isinstance(value, IntervalRange):
            if value == 0:
                return self
            value = IntervalRange.of_point(value)
        return IntervalRange(self.low + value.low, self.high + value.high, self.contain_low & value.contain_low, self.contain_high & value.contain_high)

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

    def as_integral(self, half_open: bool = False) -> 'IntervalRange':
        """
        Returns a close [n,m] interval range containing exactly all the integers in this range.
        If the half_open flag is true, return instead [n,m+1)
        """
        if self.is_empty:
            return self

        low = self.low if self.low == float('-inf') else int(math.ceil(self.low))
        if low == self.low and not self.contain_low:
            low += 1

        if half_open:
            high = self.high if self.high == float('inf') else int(math.ceil(self.high))
            if high == self.high and self.contain_high:
                high += 1
            return IntervalRange(low, high, True, False)

        high = self.high if self.high == float('inf') else int(math.floor(self.high))
        if high == self.high and not self.contain_high:
            high -= 1
        return IntervalRange[low, high]

Range = Tuple[float, float]

class InRange(OutputValidator[IntervalRange]):
    # TODO: add tests

    @classmethod
    def get_range_of(cls, elem: MapElement) -> Optional[IntervalRange]:
        in_range = next(elem.promises.output_promises(of_type=InRange), None)
        if in_range is None and isinstance(elem, MapElementConstant):
            value = elem.evaluate()
            return IntervalRange.of_point(value)
        return None if in_range is None else in_range.range

    def __init__(self, f_range: Union[IntervalRange, Tuple[float, float]]):
        self.range = f_range if isinstance(f_range, IntervalRange) else IntervalRange(*f_range)
        super().__init__(f'InRange {f_range}', context=self.range)
        self.register_validator(self._validate_constant_in_range)
        self.register_validator(self._validate_using_other_ranges)

    @staticmethod
    def consolidate_ranges(promises: OutputPromises) -> Tuple[Optional[IntervalRange], Optional[OutputPromises]]:
        promises = promises.copy()
        f_range = IntervalRange.all()
        count = 0
        in_range_promises = []
        for in_range in promises.output_promises(of_type=InRange):
            in_range_promises.append(in_range)
            count += 1
            f_range = f_range.intersection(in_range.range)
            if f_range is None:
                raise Exception(f'InRange promises collapse to an empty range')
        if count > 1:
            promises.remove_promises(in_range_promises)
            promises.add_promise(InRange(f_range))
        else:
            promises = None
        if count == 0:
            f_range = None
        return f_range, promises

    def _validate_constant_in_range(self, elem: MapElement) -> bool:
        value = elem.evaluate()
        return False if value is None else self.range.contains(value)

    def _validate_using_other_ranges(self, elem: MapElement) -> bool:
        # TODO : add test
        if self.range.is_all:
            return True
        f_range, _ = InRange.consolidate_ranges(elem.promises)
        if f_range is None:
            return False
        return self.range.contains(f_range)

    @staticmethod
    def _arithmetic_op_range_simplifier(elem: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(elem, CompositionFunction)
        if elem.function is Add:
            op = operator.add
        elif elem.function is Sub:
            op = operator.sub
        else:
            return None
        elem1, elem2 = elem.entries
        f_range1 = InRange.get_range_of(elem1)
        f_range2 = InRange.get_range_of(elem2)
        if f_range1 is None or f_range2 is None:
            return None
        interval = op(f_range1, f_range2)
        orig_interval = InRange.get_range_of(elem)
        if orig_interval is not None:
            interval = orig_interval.intersection(interval)
            if interval.contains(orig_interval):
                return None

        elem.promises.add_promise(InRange(interval))
        count, promises = InRange.consolidate_ranges(elem.promises)
        if promises is not None:
            elem.promises = promises
        return elem

def _arithmetic_op_integral_simplifier(elem: MapElement, var_dict: VarDict) -> Optional[MapElement]:
    assert isinstance(elem, CompositionFunction)
    if elem.function not in (Add, Sub, Mult):
        return None

    elem1, elem2 = elem.entries
    if elem1.has_promise(IsIntegral) and elem2.has_promise(IsIntegral):
        pass
    return None

CompositionFunction.register_class_simplifier(lambda elem, var_dict: InRange._arithmetic_op_range_simplifier(elem, var_dict))
CompositionFunction.register_class_simplifier(lambda elem, var_dict: InRange._arithmetic_op_range_simplifier(elem, var_dict))

# <editor-fold desc=" --------------- RangeCondition ---------------">

class RangeCondition(Condition, MapElementProcessor):

    def __init__(self, function: MapElement, f_range: Union[IntervalRange, Tuple[float, float]]):
        super().__init__(function.vars)
        self.function = function
        self.range = f_range if isinstance(f_range, IntervalRange) else IntervalRange(*f_range)
        self.promises.add_promise(IsCondition)

    def to_string(self, vars_to_str: Dict[Var, str]):
        return self.range.str_middle(self.function.to_string(vars_to_str))

    def __eq__(self, condition: MapElement) -> bool:
        if isinstance(condition, RangeCondition):
            return self.function == condition.function and self.range == condition.range
        return super().__eq__(condition)

    def __hash__(self) -> int:
        # TODO: Maybe have a better hash?
        return id(self)

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        return RangeCondition(self.function._call_with_dict(var_dict, func_dict), self.range)

    def process_function(self, func: MapElement, simplify: bool = True) -> MapElement:
        value = self.range.is_point
        if (value is not None) and (isinstance(self.function, Var)):
            return func({self.function: value}, simplify=simplify)
        return func

    def as_assignment(self) -> Optional[Tuple[Var, int]]:
        value = self.range.is_point
        if value is not None and value == int(value) and isinstance(self.function, Var):
            return self.function, int(value)
        return None

    # <editor-fold desc=" ======= Binary arithmetics ======= ">

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
            if condition.function.has_promise(IsIntegral):
                # Union works better for the integral version of ranges
                f_range = self.range.integral_union(condition.range)
            else:
                f_range = self.range.union(condition.range)
            return None if (f_range is None) else RangeCondition(condition.function, f_range)

        return None

    # </editor-fold>

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
    def _ranged_promise_simplifier(range_cond: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(range_cond, RangeCondition)

        function = range_cond.function
        f_range, promises = InRange.consolidate_ranges(function.promises)
        if promises is not None:
            # TODO: I don't want to change the function object itself. Consider either adding a 'copy' method
            #       to the MapElement, or instead move the promises themselves else where.
            function.promises = promises

        if f_range is None:
            return None

        if range_cond.range.contains(f_range):
            return TrueCondition

        f_range = f_range.intersection(range_cond.range)
        if f_range is None:
            return FalseCondition

        if f_range != range_cond.range or promises is not None:
            return RangeCondition(function, f_range)

        return None

    @staticmethod
    def _integral_simplifier(range_cond: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(range_cond, RangeCondition)
        if range_cond.function.has_promise(IsIntegral):
            f_range = range_cond.range.as_integral()
            if f_range != range_cond.range:
                return RangeCondition(range_cond.function, f_range)

        return None

    @staticmethod
    def _linear_combination_simplifier(element: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(element, RangeCondition)
        c1, elem1, c2, elem2 = _as_combination(element.function)

        if c1 == 1 and c2 == 0:
            # Trivial combination
            return None

        if (c2 != 0) and (elem2 is not MapElementConstant.one):
            # Too complicated combination
            return None

        # combination is c1*elem + c2.

        if c1 == 0:
            # Should have been caught in the _evaluated_simplifier, but just in case:
            return TrueCondition if element.range.contains(c2) else FalseCondition

        f_range = (element.range - c2) / c1

        return RangeCondition(elem1, f_range)

    @staticmethod
    def _in_range_arithmetic_simplification(ranged_cond: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(ranged_cond, RangeCondition)
        elem = ranged_cond.function
        if not isinstance(elem, CompositionFunction):
            return None
        if elem.function is Add:
            op = operator.add
            op1 = operator.sub
            op2 = operator.sub
        elif elem.function is Sub:
            op = operator.sub
            op1 = operator.add
            op2 = lambda e1, e2: e2-e1
        else:
            return None
        elem1, elem2 = elem.entries
        total_range = ranged_cond.range
        f_range1 = InRange.get_range_of(elem1) or IntervalRange.all()
        f_range2 = InRange.get_range_of(elem2) or IntervalRange.all()

        f_range1_updated = f_range1.intersection(op1(total_range, f_range2))
        f_range2_updated = f_range2.intersection(op2(total_range, f_range1))

        if f_range1_updated is None or f_range2_updated is None:
            return FalseCondition

        if f_range1.contains(f_range1_updated) and f_range1 != f_range1_updated:
            # TODO: this whole procedure of updating the InRange is weird. Fix it
            elem1.promises.add_promise(InRange(f_range1_updated))
            _, promises = InRange.consolidate_ranges(elem1.promises)
            if promises is not None:
                elem1.promises = promises

        if f_range2.contains(f_range2_updated) and f_range1 != f_range2_updated:
            elem2.promises.add_promise(InRange(f_range2_updated))
            _, promises = InRange.consolidate_ranges(elem2.promises)
            if promises is not None:
                elem2.promises = promises

        total_range_updated = op(f_range1_updated, f_range2_updated)
        if total_range.contains(total_range_updated):
            return RangeCondition(elem1, f_range1_updated) & RangeCondition(elem2, f_range2_updated)

        return None



    # </editor-fold>

RangeCondition.register_class_simplifier(RangeCondition._evaluated_simplifier)
RangeCondition.register_class_simplifier(RangeCondition._ranged_promise_simplifier)
RangeCondition.register_class_simplifier(RangeCondition._integral_simplifier)
RangeCondition.register_class_simplifier(RangeCondition._linear_combination_simplifier)
RangeCondition.register_class_simplifier(RangeCondition._in_range_arithmetic_simplification)

def _ranged(elem: MapElement, low: int, high: int, contains_low: bool = True, contains_high: bool = False) -> RangeCondition:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return RangeCondition(elem, IntervalRange(low, high, contains_low, contains_high))
    return NotImplemented

MapElement.__le__ = lambda self, n: _ranged(self, float('-inf'), n, False, True)
MapElement.__lt__ = lambda self, n: _ranged(self, float('-inf'), n, False, False)
MapElement.__ge__ = lambda self, n: _ranged(self, n,  float('inf'), True,  False)
MapElement.__gt__ = lambda self, n: _ranged(self, n,  float('inf'), False, False)

MapElement.__lshift__ = lambda self, n: RangeCondition(self, IntervalRange.of_point(n))

class WhereFunction:
    def __init__(self, elem: MapElement):
        self.elem = elem

    def __eq__(self, n: int):
        return RangeCondition(self.elem, IntervalRange.of_point(n))

    def __repr__(self):
        return f'Where({self.elem})'

# TODO: Refactor this entire nightmare!
MapElement.where = lambda self: WhereFunction(self)
# </editor-fold>

@always_validate_promises
class BoolVar(Var):

    def __new__(cls, var_name: str):
        return super(BoolVar, cls).__new__(cls, var_name)

    def __init__(self, name: str):
        super().__init__(name)
        self.promises.add_promise(IsIntegral)
        self.promises.add_promise(InRange(IntervalRange[0, 1]))
        self.promises.add_promise(IsCondition)