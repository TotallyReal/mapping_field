import math

from functools import cache
from typing import Optional, Union

from mapping_field.arithmetics import MultiAdd, _as_combination, _Negative
from mapping_field.conditions import (
    FalseCondition, IntersectionCondition, TrueCondition, )
from mapping_field.log_utils.tree_loggers import TreeLogger, cyan, green, red
from mapping_field.mapping_field import (
    CompositeElement, FuncDict, MapElement, MapElementConstant, MapElementProcessor,
    SimplifierContext, SimplifierOutput, Var, VarDict, class_simplifier, simplifier_context,
)
from mapping_field.property_engines import (
    PropertyByRulesEngine, is_condition, is_integral, property_rule,
)
from mapping_field.utils.processors import ProcessFailureReason

simplify_logger = TreeLogger(__name__)


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
    @cache
    def all():
        return IntervalRange(float("-inf"), float("inf"), False, False)

    @staticmethod
    @cache
    def empty():
        return IntervalRange(1, 0, False, False)

    def __init__(self, low: float, high: float, contain_low: bool = True, contain_high: bool = False):
        # TODO: make these variables frozen
        self.low = low
        self.high = high
        self.contain_low = contain_low if low != float("-inf") else False
        self.contain_high = contain_high if high != float("inf") else False
        self.is_empty = (
                (self.high < self.low) or
                (self.high == self.low and not (self.contain_low and self.contain_high)
        ))
        self.is_all = (low == float("-inf")) and (high == float("inf"))
        self.is_point: float | None = None
        if self.low == self.high and self.contain_low and self.contain_high:
            self.is_point = self.low

    def __class_getitem__(cls, params) -> "IntervalRange":
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
            return "∅"
        return f'{"[" if self.contain_low else "("}{self.low},{self.high}{"]" if self.contain_high else ")"}'

    def str_middle(self, middle: str):
        if self.is_empty:
            return "!∅!"
        if self.is_point is not None:
            return f"({middle} = {self.is_point})"
        # TODO: I can use ≤ instead of <=. However, it doesn't always looks good in terminal, and sometimes I
        #       just see the ASCII \u2264 instead.
        lower = "" if self.low == float("-inf") else (str(self.low) + ("<=" if self.contain_low else "<"))
        upper = "" if self.high == float("inf") else (("<=" if self.contain_high else "<") + str(self.high))
        return f"{lower}{middle}{upper}"

    def is_closed(self) -> bool:
        return self.contain_low and self.contain_high

    def is_open(self) -> bool:
        return not (self.contain_low or self.contain_high)

    def __eq__(self, other):
        if not isinstance(other, IntervalRange):
            return False

        if (self.is_all and other.is_all) or (self.is_empty and other.is_empty):
            return True

        if self.is_empty or other.is_empty:
            return False

        return (( self.low,  self.high,  self.contain_low,  self.contain_high) ==
                (other.low, other.high, other.contain_low, other.contain_high))

    def contains(self, other: Union[int, float, "IntervalRange"]) -> bool:
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

    # <editor-fold desc="------------------------ Boolean functions ------------------------">

    def complement(self) -> tuple["IntervalRange", "IntervalRange"]:
        if self.is_empty:
            return (self, IntervalRange.all())
        return (IntervalRange(low=float('-inf'), high=self.low, contain_low=False, contain_high=not self.contain_low) ,
                IntervalRange(low=self.high, high=float('inf'), contain_low=not self.contain_high, contain_high=False))

    def intersection(self, other: "IntervalRange") -> "IntervalRange":
        """
        Returns the intersection interval if not empty, otherwise return None.
        """
        range1 = self
        range2 = other

        if range1.is_empty or range2.is_empty:
            return IntervalRange.empty()

        low  = max(range1.low,  range2.low)
        high = min(range1.high, range2.high)
        if high < low:
            return IntervalRange.empty()

        contain_low  = range1.contains(low) and range2.contains(low)
        contain_high = range1.contains(high) and range2.contains(high)

        if high == low:
            if not contain_low:
                return IntervalRange.empty()
            return IntervalRange.of_point(low)

        return IntervalRange(low, high, contain_low, contain_high)

    def __and__(self, other: "IntervalRange") -> "IntervalRange":
        return self.intersection(other)

    def union_fill(self, other: "IntervalRange") -> "IntervalRange":
        """
        Returns the smallest interval containing this interval and the other interval
        """
        low = min(self.low, other.low)
        contain_low = self.contains(low) or other.contains(low)
        high = max(self.high, other.high)
        contain_high = self.contains(high) or other.contains(high)
        return IntervalRange(low, high, contain_low, contain_high)

    def union(self, other: "IntervalRange") -> Optional["IntervalRange"]:
        """
        Returns the union interval if it can be written as an interval, otherwise return None.
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

    def __or__(self, other: "IntervalRange") -> Optional["IntervalRange"]:
        return self.union(other)

    def integral_union(self, other: "IntervalRange") -> Optional["IntervalRange"]:
        """
        Returns the union if both intervals are considered over integral.

        Example:
            integral_union([1,3.2], [3.8, 4.5]) => [1,4]
        """
        range1 = self.as_integral(half_open=True)
        range2 = other.as_integral(half_open=True)
        result = range1.union(range2)
        return None if result is None else result.as_integral(half_open=False)

    # </editor-fold>

    # <editor-fold desc="------------------------ Arithmetic functions ------------------------">

    def __neg__(self):
        return IntervalRange(-self.high, -self.low, self.contain_high, self.contain_low)

    def __add__(self, value: Union[int, float, "IntervalRange"]) -> "IntervalRange":
        if not isinstance(value, IntervalRange):
            if value == 0:
                return self
            value = IntervalRange.of_point(value)
        if self.is_empty:
            return value
        if value.is_empty:
            return self
        return IntervalRange(
            self.low + value.low,
            self.high + value.high,
            self.contain_low & value.contain_low,
            self.contain_high & value.contain_high,
        )

    def __radd__(self, value: int | float) -> "IntervalRange":
        return self.__add__(value)

    def __sub__(self, value: int | float) -> "IntervalRange":
        return self.__add__(-value)

    def __mul__(self, value: int | float) -> "IntervalRange":
        if self.is_empty:
            return self
        if value > 0:
            return IntervalRange(self.low * value, self.high * value, self.contain_low, self.contain_high)
        if value < 0:
            return IntervalRange(self.high * value, self.low * value, self.contain_high, self.contain_low)

        # if value == 0:
        assert self.contain_low and self.contain_high
        return IntervalRange.of_point(0)

    def __rmul__(self, value: int | float) -> "IntervalRange":
        return self.__mul__(value)

    def __truediv__(self, value: int | float) -> "IntervalRange":
        assert value != 0, "DO NOT DIVIDE BY ZERO"
        return self.__mul__(1 / value)

    # </editor-fold>

    def as_integral(self, half_open: bool = False) -> "IntervalRange":
        """
        Returns a close [n,m] interval range containing exactly all the integers in this range.
        If the half_open flag is true, return instead [n,m+1)
        """
        if self.is_empty:
            return self

        low = self.low if self.low == float("-inf") else int(math.ceil(self.low))
        if low == self.low and not self.contain_low:
            low += 1

        if half_open:
            high = self.high if self.high == float("inf") else int(math.ceil(self.high))
            if high == self.high and self.contain_high:
                high += 1
            return IntervalRange(low, high, True, False)

        high = self.high if self.high == float("inf") else int(math.floor(self.high))
        if high == self.high and not self.contain_high:
            high -= 1
        return IntervalRange[low, high]


class _IntervalRangeConstructor:

    def __lshift__(self, value):
        assert isinstance(value, (int, float))
        return IntervalRange.of_point(value)

    def __lt__(self, value):
        assert isinstance(value, (int, float))
        return IntervalRange(float("-inf"), value, False, False)

    def __le__(self, value):
        assert isinstance(value, (int, float))
        return IntervalRange(float("-inf"), value, False, True)

    def __gt__(self, value):
        assert isinstance(value, (int, float))
        return IntervalRange(value, float("inf"),False, False)

    def __ge__(self, value):
        assert isinstance(value, (int, float))
        return IntervalRange(value, float("inf"),True, False)

XX = _IntervalRangeConstructor()


class RangeEngine(PropertyByRulesEngine[MapElement, SimplifierContext, IntervalRange]):

    def __init__(self):
        super().__init__()

    def compute(self, element: 'MapElement', context: 'SimplifierContext') -> IntervalRange | None:

        value = context.get_property(element, self)
        if value is not None:
            return value

        intervals = [rule(element, context) for rule in self._rules]
        intervals = [interval for interval in intervals if interval is not None]
        if len(intervals) == 0:
            return None
        final_interval = IntervalRange.all()
        for interval in intervals:
            final_interval = final_interval.intersection(interval)
        if not final_interval.is_all:
            # simplify_logger.log(f"Setting range of {red(element)} to {final_interval}")
            context.set_property(element, self, final_interval)

        return final_interval

    def combine_properties(self, prop1: IntervalRange, prop2: IntervalRange) -> IntervalRange:
        return prop1.intersection(prop2)

    def is_stronger_property(self, strong_prop: IntervalRange, weak_prop: IntervalRange) -> bool:
        return weak_prop.contains(strong_prop)

    @staticmethod
    @property_rule
    def _trivial_ranges(element: MapElement, context: SimplifierContext) -> IntervalRange | None:
        """
            5 => [5,5]
            condition => [0,1]
        """
        value = element.evaluate()
        if value is not None:
            return IntervalRange[value,value]

        if is_condition.compute(element, context):
            return IntervalRange[0,1]

        return None

    @property_rule
    def _negative_range(self, element: MapElement, context: SimplifierContext) -> IntervalRange | None:
        """
            x in [-1,10]     =>  -x in [-10,1]
        """
        if not isinstance(element, _Negative):
            return None

        interval = self.compute(element.operand, context)
        return -interval if interval is not None else None

    @property_rule
    def _multi_add_ranges(self, element: MapElement, context: SimplifierContext) -> IntervalRange | None:
        """
            x < 1 and y < 2     =>  x + y < 3
        """
        if not isinstance(element, MultiAdd):
            return None

        intervals = [self.compute(summand, context) for summand in element.operands]
        if None in intervals:
            return None
        final_interval = IntervalRange.empty()
        for interval in intervals:
            final_interval = final_interval + interval

        return final_interval

in_range = RangeEngine()
simplifier_context.register_engine(in_range)


# <editor-fold desc=" --------------- RangeCondition ---------------">


class RangeCondition(CompositeElement, MapElementProcessor):

    def __init__(self, function: MapElement, f_range: IntervalRange | tuple[float, float]):
        super().__init__(operands=[function])
        self.range = f_range if isinstance(f_range, IntervalRange) else IntervalRange(*f_range)

    @property
    def function(self) -> MapElement:
        return self.operands[0]

    @function.setter
    def function(self, value: MapElement):
        self.operands[0] = value

    def to_string(self, vars_to_str: dict[Var, str]):
        return self.range.str_middle(self.function.to_string(vars_to_str))

    def __eq__(self, condition: MapElement) -> bool:
        if isinstance(condition, RangeCondition):
            return self.function == condition.function and self.range == condition.range
        return super().__eq__(condition)

    __hash__ = MapElement.__hash__

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        return RangeCondition(self.function._call_with_dict(var_dict, func_dict), self.range)

    def process_function(self, func: MapElement, simplify: bool = True) -> MapElement:
        value = self.range.is_point
        if (value is not None) and (isinstance(self.function, Var)):
            return func({self.function: value}, simplify=simplify)
        return func

    def as_assignment(self) -> tuple[Var, int] | None:
        value = self.range.is_point
        if value is not None and value == int(value) and isinstance(self.function, Var):
            return self.function, int(value)
        return None

    # <editor-fold desc=" ======= Binary arithmetics ======= ">

    def invert(self) -> MapElement | None:
        range1, range2 = self.range.complement()
        upper = RangeCondition(self.function, range1)
        lower = RangeCondition(self.function, range2)
        return upper | lower

    def and_(self, condition: MapElement) -> MapElement | None:
        if isinstance(condition, RangeCondition) and condition.function == self.function:
            f_range = self.range.intersection(condition.range)
            return FalseCondition if (f_range is None) else RangeCondition(condition.function, f_range)

        return None

    def or_(self, condition: MapElement) -> MapElement | None:
        simplify_logger.log(f"Computing 'or' of {red(self)} with {red(condition)} [{cyan(self.__class__.__name__)}]")
        if isinstance(condition, RangeCondition) and condition.function == self.function:
            if is_integral.compute(condition.function, simplifier_context):
                # Union works better for the integral version of ranges
                simplify_logger.log(f"Trying to combine integral ranges")
                f_range = self.range.integral_union(condition.range)
            else:
                simplify_logger.log(f"Trying to combine ranges")
                f_range = self.range.union(condition.range)
            return None if (f_range is None) else RangeCondition(condition.function, f_range)

        if isinstance(condition, MapElementProcessor):
            processed_function = condition.process_function(self.function)
            value = processed_function.evaluate()
            if value is None:
                return None
            simplify_logger.log(f'Condition {red(condition)} implies {green(self.function)}={green(value)}')
            if self.range.contains(value):
                return self

            # See if it extends the range to an interval.
            interval = IntervalRange.of_point(value)
            if is_integral.compute(self.function, simplifier_context):
                interval = self.range.integral_union(interval)
            else:
                interval = self.range.union(interval)
            if interval is None:
                return None

            # See if it exactly what was missing:
            at_value_cond = (self.function << value)
            simplify_logger.log(f'Check if {red(condition)} == {red(at_value_cond)}')
            at_value_cond = at_value_cond.simplify()
            if at_value_cond == condition:
                return RangeCondition(self.function, interval)

        return None

    # </editor-fold>

    # <editor-fold desc=" ======= Simplifiers ======= ">

    def _simplify_with_var_values(self) -> SimplifierOutput:
        if self.range.is_empty:
            return FalseCondition
        if self.range.is_all:
            return TrueCondition

        cur_range = in_range.compute(self.function, simplifier_context)
        if cur_range is None:
            return ProcessFailureReason('No existing range to intersect with', trivial=True)
        if self.range.contains(cur_range):
            return TrueCondition
        if cur_range.contains(self.range):
            return ProcessFailureReason('New range is smaller than existing range', trivial=False)

        f_range = cur_range.intersection(self.range)
        if f_range.is_empty:
            return FalseCondition

        return RangeCondition(self.function, f_range)

    @class_simplifier
    @staticmethod
    def _linear_simplifier(element: MapElement) -> SimplifierOutput:
        """
            c*x + d < r     =>      x < (r-d)/c     (for c>0)
        """
        assert isinstance(element, RangeCondition)
        c1, elem1, c2, elem2 = _as_combination(element.function)

        if c1 == 1 and c2 == 0:
            return ProcessFailureReason("Trivial combination", trivial=True)

        if (c2 != 0) and (elem2 is not MapElementConstant.one):
            # Too complicated combination
            return None

        # combination is c1*elem + c2.

        if c1 == 0:
            # Should have been caught in the _evaluated_simplifier, but just in case:
            return TrueCondition if element.range.contains(c2) else FalseCondition

        f_range = (element.range - c2) / c1

        return RangeCondition(elem1, f_range)


    @class_simplifier
    @staticmethod
    def _integral_simplifier(range_cond: MapElement) -> SimplifierOutput:
        """
            (f integral) < 5.4      =>      (f integral) <= 5
        """
        assert isinstance(range_cond, RangeCondition)
        if is_integral.compute(range_cond.function, simplifier_context):
            f_range = range_cond.range.as_integral()
            if f_range != range_cond.range:
                return RangeCondition(range_cond.function, f_range)

        return ProcessFailureReason("Function is not Integral", trivial=False)

    @class_simplifier
    @staticmethod
    def _sum_equality_to_summands_equality_simplifier(ranged_cond: MapElement) -> SimplifierOutput:
        """
            if x_i in [a_i, b_i], then
                    sum x_i = sum a_i   =>  x_i = a_i
        """
        assert isinstance(ranged_cond, RangeCondition)
        elem = ranged_cond.function
        if not isinstance(elem, MultiAdd):
            return ProcessFailureReason('Only works for Addition', trivial=True)
        sum_range = ranged_cond.range
        value = sum_range.is_point
        if value is None:
            return ProcessFailureReason('Only applicable for sum x_i = point ranges', trivial=True)

        ranges = [in_range.compute(summand, simplifier_context) for summand in elem.operands]
        if None in ranges:
            return ProcessFailureReason('Not all summands have range condition', trivial=True)

        if value == sum(f_range.low for f_range in ranges):
            return IntersectionCondition(
                [RangeCondition(summand, IntervalRange.of_point(f_range.low)) for summand, f_range in zip(elem.operands, ranges)],
            )

        if value == sum(f_range.high for f_range in ranges):
            return IntersectionCondition(
                [RangeCondition(summand, IntervalRange.of_point(f_range.high)) for summand, f_range in zip(elem.operands, ranges)],
            )

        return None

    @class_simplifier
    @staticmethod
    def _condition_in_range_simplifier(element: MapElement) -> SimplifierOutput:
        """
                cond == 1     =>      cond
        """
        # TODO: add tests.
        #       I don't like this step too much. If I start with (x << 1) for bool var, it will become just x.
        #       However, if I now try to set x = 1, instead of getting TrueCondition, I will get 1. And while they are
        #       the same, it is easier to think about them (and view them on screen) differently.
        assert isinstance(element, RangeCondition)
        if not is_condition.compute(element.function, simplifier_context):
            return ProcessFailureReason("Only applicable for ranges on conditions")
        if isinstance(element.function, Var):
            return None
        if element.range == IntervalRange.of_point(1):
            return element.function
        if element.range == IntervalRange.of_point(0):
            return ~element.function
        return None

    # </editor-fold>

is_condition.add_auto_class(RangeCondition)

def _ranged(
    elem: MapElement, low: int, high: int, contains_low: bool = True, contains_high: bool = False
) -> RangeCondition:
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return RangeCondition(elem, IntervalRange(low, high, contains_low, contains_high))
    return NotImplemented


MapElement.__le__ = lambda self, n: _ranged(self, float("-inf"), n, False, True)
MapElement.__lt__ = lambda self, n: _ranged(self, float("-inf"), n, False, False)
MapElement.__ge__ = lambda self, n: _ranged(self, n, float("inf"), True, False)
MapElement.__gt__ = lambda self, n: _ranged(self, n, float("inf"), False, False)

MapElement.__lshift__ = lambda self, n: RangeCondition(self, IntervalRange.of_point(n))


class WhereFunction:
    def __init__(self, elem: MapElement):
        self.elem = elem

    def __eq__(self, n: int):
        return RangeCondition(self.elem, IntervalRange.of_point(n))

    def __repr__(self):
        return f"Where({self.elem})"


# TODO: Refactor this entire nightmare!
MapElement.where = lambda self: WhereFunction(self)
# </editor-fold>


