import math
import operator

from abc import abstractmethod
from functools import cache
from typing import Optional, Union

from mapping_field.arithmetics import MultiAdd, _Add, _as_combination, _Mult, _Negative
from mapping_field.conditions import (
    BinaryCondition, FalseCondition, IntersectionCondition, TrueCondition, UnionCondition,
)
from mapping_field.log_utils.tree_loggers import TreeLogger, green, red
from mapping_field.mapping_field import (
    CompositeElement, FuncDict, MapElement, MapElementConstant, MapElementProcessor, OutputPromises,
    OutputValidator, SimplifierOutput, Var, VarDict, class_simplifier, simplifier_context, SimplifierContext,
)
from mapping_field.property_engines import is_condition, is_integral, PropertyByRulesEngine, property_rule
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

    def is_closed(self) -> bool:
        return self.contain_low and self.contain_high

    def is_open(self) -> bool:
        return ~self.contain_low and ~self.contain_high

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

    def __eq__(self, other):
        if not isinstance(other, IntervalRange):
            return False

        if not self.is_empty and not other.is_empty:
            return (( self.low,  self.high,  self.contain_low,  self.contain_high) ==
                    (other.low, other.high, other.contain_low, other.contain_high))

        return self.is_empty == other.is_empty

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

    def complement(self) -> tuple["IntervalRange", "IntervalRange"]:
        if self.is_empty:
            return (self, IntervalRange.all())
        return (IntervalRange(low=float('-inf'), high=self.low, contain_low=False, contain_high=not self.contain_low) ,
                IntervalRange(low=self.high, high=float('inf'), contain_low=not self.contain_high, contain_high=False))

    def intersection(self, other: "IntervalRange") -> Optional["IntervalRange"]:
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

    def integral_union(self, other: "IntervalRange") -> Optional["IntervalRange"]:
        range1 = self.as_integral(half_open=True)
        range2 = other.as_integral(half_open=True)
        result = range1.union(range2)
        return None if result is None else result.as_integral(half_open=False)

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


class Ranged:
    @abstractmethod
    def get_range(self) -> IntervalRange | None:
        raise NotImplementedError()

class RangeEngine(PropertyByRulesEngine[IntervalRange]):

    def __init__(self):
        super().__init__()

    def compute(self, element: 'MapElement', context: 'SimplifierContext') -> IntervalRange | None:
        in_range = next(element.promises.output_promises(of_type=InRange), None)
        if in_range is not None:
            return in_range.range

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

    @property_rule
    def _negative_range(self, element: MapElement, context: SimplifierContext) -> IntervalRange | None:
        """
            x in [-1,10]     =>  -x in [-10,1]
        """
        if not isinstance(element, _Negative):
            return None

        interval = self.compute(element.operand, context)
        return -interval if interval is not None else None

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

        if isinstance(element, Ranged):
            return element.get_range()

        if is_condition.compute(element, context):
            return IntervalRange[0,1]

        return None

in_range = RangeEngine()
simplifier_context.register_engine(in_range)

class InRange(OutputValidator[IntervalRange]):
    # TODO: add tests

    @classmethod
    def get_range_of(cls, elem: MapElement) -> IntervalRange | None:
        in_range = next(elem.promises.output_promises(of_type=InRange), None)
        if in_range is not None:
            return in_range.range

        if isinstance(elem, MapElementConstant):
            value = elem.evaluate()
            return IntervalRange.of_point(value)
        if is_condition.compute(elem, simplifier_context):
            return IntervalRange[0,1]
        if isinstance(elem, Ranged):
            return elem.get_range()

        return None

    def __init__(self, f_range: IntervalRange | tuple[float, float]):
        self.range = f_range if isinstance(f_range, IntervalRange) else IntervalRange(*f_range)
        c = self.range.is_point
        name = f"Equal {c}" if c is not None else f"InRange {f_range}"
        super().__init__(name, context=self.range)
        self.register_validator(self._validate_constant_in_range)
        self.register_validator(self._validate_using_other_ranges)
        self.register_validator(self.contain_validate)

    def contain_validate(self, elem: MapElement) -> bool | None:
        elem_range = in_range.compute(elem, simplifier_context)
        if elem_range is None:
            return None
        if self.range.contains(elem_range):
            return True
        return None

    # @staticmethod
    # def consolidate_ranges(promises: OutputPromises) -> tuple[IntervalRange | None, OutputPromises | None]:
    #     promises = promises.copy()
    #     f_range = IntervalRange.all()
    #     count = 0
    #     in_range_promises = []
    #     for in_range in promises.output_promises(of_type=InRange):
    #         in_range_promises.append(in_range)
    #         count += 1
    #         f_range = f_range.intersection(in_range.range)
    #         if f_range is None:
    #             raise Exception(f"InRange promises collapse to an empty range")
    #     if count > 1:
    #         promises.remove_promises(in_range_promises)
    #         promises.add_promise(InRange(f_range))
    #     else:
    #         promises = None
    #     if count == 0:
    #         f_range = None
    #     return f_range, promises

    def _validate_constant_in_range(self, elem: MapElement) -> bool | None:
        value = elem.evaluate()
        if value is None:
            return None
        return self.range.contains(value)

    # def _validate_using_other_ranges(self, elem: MapElement) -> bool | None:
    #     # TODO : add test
    #     if self.range.is_all:
    #         return True
    #     f_range, _ = InRange.consolidate_ranges(elem.promises)
    #     if f_range is None:
    #         return None
    #     return self.range.contains(f_range)
    #
    # @staticmethod
    # def _negation_range_simplifier(elem: MapElement) -> SimplifierOutput:
    #     assert isinstance(elem, _Negative)
    #
    #     interval = in_range.compute(elem.operand, simplifier_context)
    #     if interval is None:
    #         return ProcessFailureReason('Operand does not have a range', trivial=True)
    #     interval = -interval
    #
    #     orig_interval = in_range.compute(elem, simplifier_context)
    #     if orig_interval is not None and interval.contains(orig_interval):
    #         return ProcessFailureReason('The current range is already smaller than the one from the operand', trivial=True)
    #
    #     # TODO: need to create a new element
    #     elem.promises.add_promise(InRange(interval))
    #     count, promises = InRange.consolidate_ranges(elem.promises)
    #     simplify_logger.log(f"Added range {green(interval)} to {green(elem)}")
    #     if promises is not None:
    #         elem.promises = promises
    #     return elem

    # @staticmethod
    # def _arithmetic_op_range_simplifier(elem: MapElement) -> SimplifierOutput:
    #     """
    #         x + y = 2   =>  (x=1) & (y=1)       [if true...]
    #     """
    #     assert isinstance(elem, _Add)
    #     op = operator.add
    #
    #     elem1, elem2 = elem.operands
    #     f_range1 = in_range.compute(elem1, simplifier_context)
    #     f_range2 = in_range.compute(elem2, simplifier_context)
    #     if f_range1 is None or f_range2 is None:
    #         return None
    #     interval = op(f_range1, f_range2)
    #     orig_interval = in_range.compute(elem, simplifier_context)
    #     if orig_interval is not None:
    #         interval = orig_interval.intersection(interval)
    #         if interval.contains(orig_interval):
    #             return None
    #
    #     elem.promises.add_promise(InRange(interval))
    #     count, promises = InRange.consolidate_ranges(elem.promises)
    #     simplify_logger.log(f"Added range {green(interval)} to {green(elem)}")
    #     if promises is not None:
    #         elem.promises = promises
    #     return elem


# _Add.register_class_simplifier(InRange._arithmetic_op_range_simplifier)
# _Negative.register_class_simplifier(InRange._negation_range_simplifier)


# <editor-fold desc=" --------------- RangeCondition ---------------">


class RangeCondition(CompositeElement, MapElementProcessor):

    def __init__(self, function: MapElement, f_range: IntervalRange | tuple[float, float]):
        super().__init__(operands=[function], output_properties={})
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
        if isinstance(condition, RangeCondition) and condition.function == self.function:
            if is_integral.compute(condition.function, simplifier_context):
                # Union works better for the integral version of ranges
                f_range = self.range.integral_union(condition.range)
            else:
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
            at_value_cond = at_value_cond.simplify2()
            if at_value_cond == condition:
                return RangeCondition(self.function, interval)

        return None

    # </editor-fold>

    # <editor-fold desc=" ======= Simplifiers ======= ">

    def _simplify_with_var_values2(self) -> SimplifierOutput:
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
        if is_bool_var(element.function): # TODO: Only Var?
            return None
        if element.range == IntervalRange.of_point(1):
            return element.function
        if element.range == IntervalRange.of_point(0):
            return ~element.function
        return None

    # @class_simplifier
    # @staticmethod
    # def _ranged_promise_simplifier(range_cond: MapElement) -> SimplifierOutput:
    #     """
    #     Consolidate ranges on a function
    #     """
    #     assert isinstance(range_cond, RangeCondition)
    #
    #     function = range_cond.function
    #     f_range, promises = InRange.consolidate_ranges(function.promises)
    #     if f_range is None:
    #         return ProcessFailureReason("Function has no range", trivial=True)
    #     if promises is not None:
    #         # TODO: I don't want to change the function object itself. Consider either adding a 'copy' method
    #         #       to the MapElement, or instead move the promises themselves else where.
    #         function.promises = promises
    #
    #     if range_cond.range.contains(f_range):
    #         return TrueCondition
    #
    #     f_range = f_range.intersection(range_cond.range)
    #     if f_range is None:
    #         return FalseCondition
    #
    #     if f_range != range_cond.range or promises is not None:
    #         return RangeCondition(function, f_range)
    #
    #     return None

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
    def _linear_combination_simplifier(element: MapElement) -> SimplifierOutput:
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
    def _negation_in_range_simplification(ranged_cond: MapElement) -> SimplifierOutput:
        """
            -x in I     =>      x in -I
        """
        assert isinstance(ranged_cond, RangeCondition)
        if not isinstance(ranged_cond.function, _Negative):
            return ProcessFailureReason('Only works for Negation', trivial=True)
        return RangeCondition(ranged_cond.function.operand, -ranged_cond.range)

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


def BoolVar(name: str) -> Var:
    return Var(name=name, output_properties={is_condition: True})

def is_bool_var(v: MapElement) -> bool:
    if not isinstance(v, Var):
        return False
    if is_condition.compute(v, simplifier_context):
        return True

    f_range = in_range.compute(v, simplifier_context)

    return (f_range is not None) and IntervalRange[0,1].contains(f_range) and is_integral.compute(v, simplifier_context)


# class BoolVar(Var):
#
#     def __init__(self, name: str):
#         super().__init__(name)
#         self.promises.add_promise(IsIntegral)
#         in_range.add_range(self, IntervalRange[0, 1])
#         # self.promises.add_promise(InRange(IntervalRange[0, 1]))
#         self.promises.add_promise(IsCondition)

def two_bool_vars_simplifier(elem: MapElement) -> SimplifierOutput:
    # TODO: make sure that I don't call has_promise for an element that I am trying to simplify, since it might
    #       call simplify inside it, and then we ar off to the infinite loop races.
    # if not is_condition.compute(elem, simplifier_context):
    if not is_condition.compute(elem, simplifier_context):
        return ProcessFailureReason("Not a Condition", trivial=True)
    # if len(var_dict) > 0:
    #     return ProcessFailureReason("Only applicable with no var_dict", trivial=True)
    if len(elem.vars) > 2 or (not all(is_bool_var(v) for v in elem.vars)):
        return ProcessFailureReason("Only applicable with at most 2 bool vars", trivial=True)

    if len(elem.vars) == 1:
        v = elem.vars[0]
        simplify_logger.log(f"Looking for simpler condition on {red(v)}")
        if elem in (v, (v << 0), (v << 1)):
            return None
        # TODO: The following two calls are problematics. They can generate composition function with 'elem'
        #       as the top function, so when we call simplify on it, it tries to simplify the top function
        #       by itself, which can loop back here.
        value0 = elem({v: 0}).simplify2()
        value1 = elem({v: 1}).simplify2()
        if not (isinstance(value0, BinaryCondition) and isinstance(value1, BinaryCondition)):
            simplify_logger.log(red(f"The values {value0}, {value1} should be binary."))
            return None
        if value0 is value1 is TrueCondition:
            return TrueCondition
        if value0 is value1 is FalseCondition:
            return FalseCondition
        return (v << 0) if value0 is TrueCondition else (v << 1)

    if len(elem.vars) == 2:
        x, y = elem.vars
        simplify_logger.log(f"Looking for simpler condition on {red(x)}, {red(y)}")
        assignments = [(0, 0), (0, 1), (1, 0), (1, 1)]
        values = [[elem({x: x0, y: y0}).simplify2() for y0 in (0, 1)] for x0 in (0, 1)]
        if not all(isinstance(value, BinaryCondition) for value in values[0] + values[1]):
            values_str = ", ".join(str(value) for value in values[0] + values[1])
            simplify_logger.log(red(f"The values {values_str} should be binary."))
            return None

        count_true = sum(value is TrueCondition for value in values[0] + values[1])
        if count_true == 0:
            return FalseCondition
        if count_true == 1:
            for x0, y0 in assignments:
                if values[x0][y0] is TrueCondition:
                    result = IntersectionCondition([x << x0, y << y0], simplified=True)
                    # Don't call (x << x0) & (y << y0) since it automatically calls to simplify, which can
                    # cause an infinite loop
                    # TODO: should I split to structure simplified and promise simplified?
                    return result if elem != result else None
        if count_true == 2:
            if values[0][0] == values[1][1]:
                # TODO: add this when I know how to compare elements like (x+y = 1, x-y=0)
                return None
            v = y if (values[0][0] == values[1][0]) else x
            value = 0 if values[0][0] is TrueCondition else 1
            result = v << value
            return result
        if count_true == 3:
            for x0, y0 in assignments:
                if values[x0][y0] is FalseCondition:
                    result = UnionCondition([x << 1 - x0, y << 1 - y0], simplified=True)
                    return result if elem != result else None
        if count_true == 4:
            return TrueCondition

        # Should never get here...
        raise Exception("Learn how to count to 4...")

    return None

def mult_binary_assignment_by_numbers(element: MapElement) -> SimplifierOutput:
    """
    change multiplications of (x << 1) * c into c * x for boolean variables x.
    """
    assert isinstance(element, _Mult)
    operands = element.operands
    value0 = operands[0].evaluate()
    value1 = operands[1].evaluate()
    if (value0 is None) + (value1 is None) != 1:
        return ProcessFailureReason("Exactly one of the factors must by a constant value", trivial=True)

    value = value0 or value1
    elem = operands[1] if value1 is None else operands[0]
    if isinstance(elem, RangeCondition) and is_bool_var(elem.function):
        if elem.range.is_point == 1 and value != 1:
            return value * elem.function
    return None

MapElement._simplifier.register_processor(two_bool_vars_simplifier)

_Mult.register_class_simplifier(mult_binary_assignment_by_numbers)