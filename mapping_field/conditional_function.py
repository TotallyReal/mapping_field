import operator

from typing import Dict, List, Optional, Tuple, Union

from mapping_field.arithmetics import _Mult
from mapping_field.conditions import FalseCondition, TrueCondition, UnionCondition
from mapping_field.field import ExtElement
from mapping_field.log_utils.tree_loggers import TreeLogger, green, red, yellow
from mapping_field.mapping_field import (
    CompositeElement, MapElement, MapElementConstant, MapElementProcessor, OutputValidator, Var,
    convert_to_map, params_to_maps,
)
from mapping_field.promises import IsCondition, IsIntegral
from mapping_field.ranged_condition import InRange, IntervalRange, RangeCondition, Ranged
from mapping_field.utils.processors import ProcessFailureReason

simplify_logger = TreeLogger(__name__)


class SingleRegion(CompositeElement, Ranged):

    @classmethod
    def of(cls, element):
        if isinstance(element, SingleRegion):
            return element
        if isinstance(element, tuple) and len(element)==2:
            condition = convert_to_map(element[0])
            function = convert_to_map(element[1])
            if (
                    isinstance(function, MapElement) and
                    isinstance(condition, MapElement) and
                    condition.has_promise(IsCondition)
            ):
                return SingleRegion(condition, function)

        return None


    def __init__(self, condition: MapElement, function: MapElement):
        assert condition.has_promise(IsCondition)
        # TODO: Transfer promises from the function to this SingleRegion?
        super().__init__(operands=[condition, function])

    @property
    def condition(self) -> MapElement:
        return self.operands[0]

    @condition.setter
    def condition(self, value: MapElement):
        self.operands[0] = value

    @property
    def function(self) -> MapElement:
        return self.operands[1]

    @function.setter
    def function(self, value: MapElement):
        self.operands[1] = value

    def to_string(self, vars_to_str: Dict[Var, str]):
        # TODO: fix this printing function
        return f" {self.condition.to_string(vars_to_str)} -> {self.function.to_string(vars_to_str)} "

    def __iter__(self):
        yield self.condition
        yield self.function

    def _simplify_with_var_values2(self) -> Optional[MapElement]:
        if not isinstance(self.condition, MapElementProcessor):
            return None

        # TODO: Use a process_function process like in map elements instead
        function = self.function
        simplified_func = self.condition.process_function(function)
        if simplified_func is function:
            return None

        function = simplified_func
        function = function._simplify2() or function

        return SingleRegion(self.condition, function)


class ConditionalFunction(CompositeElement, Ranged):
    """
    A conditional function of the form:
       1_(cond_1) * f_1 + 1_(cond_2) * f_2 + ... + 1_(cond_n) * f_n

    Working under the assumption that the conditions do not intersect, and cover the whole space, namely
    the form a decomposition of the whole space.
    """

    @staticmethod
    def always(map: MapElement):
        return ConditionalFunction([(TrueCondition, map)])

    def __init__(self, regions: List[Union[Tuple[MapElement, MapElement], SingleRegion]]):
        true_regions = [SingleRegion.of(region) for region in regions]
        assert None not in true_regions, f'Could not convert the regions {regions} into SingleRegions.'

        super().__init__(operands=true_regions)

    @property
    def regions(self) -> List[SingleRegion]:
        return self.operands

    @regions.setter
    def regions(self, value: List[MapElement]):
        assert all(isinstance(region, SingleRegion) for region in value)
        self.operands = value


    def to_string(self, vars_to_str: Dict[Var, str]):
        # TODO: fix this printing function
        inner_str = " ; ".join([region.to_string(vars_to_str) for region in self.regions])
        return f"[{inner_str}]"

    #
    #     def pretty_str(self):
    #         return '\n  @@       +      @@  \n'.join([
    #             f'Given:  \n{condition.pretty_str() if isinstance(condition, _ListCondition) else repr(condition)} \n -->  {repr(map)}'
    #             for condition, map in self.regions
    #         ])

    def evaluate(self) -> Optional[ExtElement]:
        values = [region.function.evaluate() for region in self.regions]
        if len(values) == 0:
            raise Exception("Conditional Map should not be empty")
        if values[0] is None:
            return None
        return values[0] if all([values[0] == v for v in values]) else None

    @params_to_maps
    def __eq__(self, other: MapElement) -> bool:
        if isinstance(other, ConditionalFunction):
            return self._op(other, operator.sub).simplify2().is_zero()

        for cond, func in self.regions:
            if isinstance(cond, MapElementProcessor):
                if cond.process_function(func) != cond.process_function(other):
                    return False
            else:
                if func != other:
                    return False

        return True

    def get_range(self) -> Optional[IntervalRange]:
        interval = IntervalRange.empty()
        for region in self.regions:
            next_interval = InRange.get_range_of(region)
            if next_interval is None:
                return None
            interval = interval.union_fill(next_interval)
        return interval

    # <editor-fold desc=" ------------------------ arithmetics ------------------------">

    def _op(self, other: MapElement, op_func) -> "ConditionalFunction":
        if isinstance(other, int):
            other = MapElementConstant(other)
        if not isinstance(other, ConditionalFunction):
            other = ConditionalFunction.always(other)
        regions: List[Tuple[MapElement, MapElement]] = []
        for cond1, elem1 in self.regions:
            for cond2, elem2 in other.regions:
                simplify_logger.log(f"check if the regions {red(cond1)}, {red(cond2)} intersect.")
                cond_prod = (cond1 & cond2).simplify2()
                if cond_prod is not FalseCondition:
                    simplify_logger.log(
                        f'Regions intersect - apply "{yellow(op_func.__name__)}" on the functions {red(elem1)}, {red(elem2)}.'
                    )
                    regions.append((cond_prod, op_func(elem1, elem2)))
        return ConditionalFunction(regions).simplify2()

    def add(self, other: MapElement) -> "ConditionalFunction":
        return self._op(other, operator.add)

    def radd(self, other: MapElement) -> "ConditionalFunction":
        return self._op(other, operator.add)

    def mul(self, other: MapElement) -> "ConditionalFunction":
        return self._op(other, operator.mul)

    def rmul(self, other: MapElement) -> "ConditionalFunction":
        return self._op(other, operator.mul)

    def sub(self, other: MapElement) -> "ConditionalFunction":
        if other == 0:
            return self
        return self._op(other, operator.sub)

    def rsub(self, other: MapElement) -> "ConditionalFunction":
        return ConditionalFunction.always(other)._op(self, operator.sub)

    def div(self, other: MapElement) -> "ConditionalFunction":
        return self._op(other, operator.truediv)

    def rdiv(self, other: MapElement) -> "ConditionalFunction":
        return ConditionalFunction.always(other)._op(self, operator.truediv)

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Simplifiers and Validators ------------------------ ">

    def _simplify_with_var_values2(self) -> Optional[MapElement]:
        # TODO: combine this simplification with the _ListCondition simplification for a general simplification of
        #       a commutative associative binary function.

        def combinable(
            condition1: MapElement, elem1: MapElement, condition2: MapElement, elem2: MapElement
        ) -> Optional[MapElement]:
            """
            Look for a single MapFunctions elem such that :
                elem(condition1) = elem1 ,
                elem(condition2) = elem2
            """
            if elem1 == elem2:
                return elem1

            if isinstance(condition1, MapElementProcessor):
                # TODO: use __call__(condition1) instead?
                simplify_logger.log(f"Computing {red(elem2)} at {red(condition1)}")
                processed_elem2 = condition1.process_function(elem2)
                if processed_elem2 == elem1:
                    return elem2

            if isinstance(condition2, MapElementProcessor):
                # TODO: use __call__(condition1) instead?
                simplify_logger.log(f"Computing {red(elem1)} at {red(condition2)}")
                processed_elem1 = condition2.process_function(elem1)
                if processed_elem1 == elem2:
                    return elem1

            return None

        regions = []
        is_simpler = False
        for region in self.regions:

            simplified_region = region._simplify2()
            if simplified_region is not None:
                is_simpler = True
            region = simplified_region or region
            if region.condition is FalseCondition:
                continue

            for i, prev_region in enumerate(regions):
                simplify_logger.log(f"Trying to combine {red(region)} and {red(prev_region)}.")
                comb_elem = combinable(prev_region.condition, prev_region.function, region.condition, region.function)
                if comb_elem is not None:
                    simplify_logger.log(f"They are combinable under the function {green(comb_elem)}")
                    is_simpler = True
                    condition_union = prev_region.condition | region.condition
                    combined_region = SingleRegion(condition=condition_union, function=comb_elem)
                    combined_region = combined_region.simplify2()
                    regions[i] = combined_region
                    break
            else:
                regions.append(region)

        # TODO: The conditions in a conditional function should cover the whole space, so a single region
        #       must always have a TrueCondition. However, it is not always true that it is easy to check
        #       that the condition is true. Should I keep this check here or not?
        if len(regions) == 1:  # and regions[0][0] is TrueCondition:
            return regions[0].function

        return ConditionalFunction(regions) if is_simpler else None

    @staticmethod
    def mult_condition_by_element(element: MapElement) -> Optional[MapElement]:
        assert isinstance(element, _Mult)
        a, b = element.operands
        a_is_cond = a.has_promise(IsCondition)
        b_is_cond = b.has_promise(IsCondition)
        if b_is_cond:
            a, b = b, a
        if not a_is_cond or b_is_cond:
            return None

        if b_is_cond:
            a, b = b, a
        if isinstance(a, (Var, MapElementConstant)):
            return None
        return ConditionalFunction([(a, b), (~a, MapElementConstant.zero)])

    @staticmethod
    def new_assignment_simplify(ranged_cond: MapElement) -> Optional[MapElement]:
        assert isinstance(ranged_cond, RangeCondition)
        cond_function = ranged_cond.function
        if not isinstance(cond_function, ConditionalFunction):
            return None

        f_range = ranged_cond.range
        return UnionCondition([condition & RangeCondition(func, f_range) for condition, func in cond_function.regions])

    @staticmethod
    def _bool_var_simplifier(
        map_elem: MapElement
    ) -> Optional[Union[MapElement, ProcessFailureReason]]:
        assert isinstance(map_elem, ConditionalFunction)

        if len(map_elem.regions) != 2:
            return ProcessFailureReason("Only works for two regions", trivial=True)

        cond1, func1 = map_elem.regions[0]
        cond2, func2 = map_elem.regions[1]
        value1 = func1.evaluate()
        value2 = func2.evaluate()
        if value1 is None or value2 is None:
            return None

        simplify_logger.log(f'Verify that the 2 regions combine to the whole space.')
        if (cond1 | cond2).simplify2() is not TrueCondition:
            return ProcessFailureReason(
                "Conditional Function on two regions should have complement conditions, "
                "instead got {red(cond1)}, {red(cond2)}",
                trivial=False
            )

        return value2 + (value1 - value2) * cond1

        # if not (isinstance(cond1, RangeCondition) and isinstance(cond2, RangeCondition)):
        #     return None
        # assign1 = cond1.as_assignment()
        # assign2 = cond2.as_assignment()
        # if assign1 is None or assign2 is None:
        #     return None
        #
        # v1 = assign1[0]
        # v2 = assign2[0]
        # if not (isinstance(v1, BoolVar) and v1 is v2):
        #     return None
        #
        # assigned_value1 = assign1[1]
        # assigned_value2 = assign2[1]
        #
        # if (assigned_value1, assigned_value2) == (0, 1):
        #     return (value1 + (value2 - value1) * v1).simplify2()
        #
        # if (assigned_value1, assigned_value2) == (1, 0):
        #     return (value2 + (value1 - value2) * v1).simplify2()
        #
        # raise Exception(
        #     f"The assigned values should be 0 and 1, but instead got {assigned_value1} and {assigned_value2}"
        # )

    @staticmethod
    def promise_validate_conditional_function(validator: OutputValidator, elem: MapElement) -> Optional[bool]:
        # TODO: This is not precise, because a function's promise can depend on where it is defined,
        #       but let's keep it simple for now...
        if not isinstance(elem, ConditionalFunction):
            return None
        validations = [function.has_promise(validator) for _, function in elem.regions]
        if any([validation is False for validation in validations]):
            return False
        if any([validation is None for validation in validations]):
            return None
        return True

    # </editor-fold>


_Mult.register_class_simplifier(ConditionalFunction.mult_condition_by_element)
RangeCondition.register_class_simplifier(ConditionalFunction.new_assignment_simplify)
ConditionalFunction.register_class_simplifier(ConditionalFunction._bool_var_simplifier)

# TODO: Make it work for any validator in the future
IsIntegral.register_validator(lambda elem: ConditionalFunction.promise_validate_conditional_function(IsIntegral, elem))
# TODO: Don't think of ConditionalFunction as a condition for now
# IsCondition.register_validator(lambda elem: promise_validate_conditional_function(IsCondition, elem))


def ReLU(map_elem: MapElement) -> MapElement:
    f_range = InRange.get_range_of(map_elem)
    if f_range is not None and f_range.low >= 0:
        return map_elem
    zero = MapElementConstant.zero
    if isinstance(map_elem, ConditionalFunction):
        regions = []
        for condition, func in map_elem.regions:
            non_negative = func >= 0
            if (func >= 0).simplify2() is TrueCondition:
                # Make your and my life a little bit simpler
                regions.append((condition, func))
            elif (func <= 0).simplify2() is TrueCondition:
                regions.append((condition, zero))
            else:
                regions.append((condition & (func > 0), func))
                regions.append((condition & (func <= 0), zero))
        regions = [(cond, func) for cond, func in regions if FalseCondition != cond]
        return ConditionalFunction(regions)
    return ConditionalFunction([
        ((map_elem > 0), map_elem),
        ((map_elem <= 0), zero)
    ])