import operator

from typing import Dict, List, Optional, Tuple, Union

from mapping_field.arithmetics import Mult
from mapping_field.conditions import FalseCondition, TrueCondition, UnionCondition
from mapping_field.field import ExtElement
from mapping_field.log_utils.tree_loggers import TreeLogger, green, red, yellow
from mapping_field.mapping_field import (
    FuncDict, MapElement, MapElementConstant, MapElementProcessor, OutputValidator, Var, VarDict,
    convert_to_map, params_to_maps,
)
from mapping_field.processors import ProcessFailureReason
from mapping_field.promises import IsCondition, IsIntegral
from mapping_field.ranged_condition import BoolVar, InRange, RangeCondition

simplify_logger = TreeLogger(__name__)


class SingleRegion(MapElement):

    def __init__(self, condition: MapElement, function: MapElement):
        assert condition.has_promise(IsCondition)
        # TODO: Transfer promises from the function to this SingleRegion?
        super().__init__(list(set(condition.vars + function.vars)))
        self.condition = condition
        self.function = function

    def to_string(self, vars_to_str: Dict[Var, str]):
        # TODO: fix this printing function
        return f" {repr(self.condition)} -> {repr(self.function)} "

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        return SingleRegion(
            self.condition._call_with_dict(var_dict, func_dict),
            self.function._call_with_dict(var_dict, func_dict)
        )

    def __iter__(self):
        yield self.condition
        yield self.function

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        is_simpler = False
        simplified_condition = self.condition._simplify2(var_dict)
        simplified_func = self.function._simplify2(var_dict)
        if (simplified_condition or simplified_func) is not None:
            is_simpler = True
        condition = simplified_condition or self.condition
        function = simplified_func or self.function

        if isinstance(condition, MapElementProcessor):
            # TODO: Use a process_function process like in map elements instead
            simplified_func = condition.process_function(function)
            is_simpler |= (simplified_func is not function)
            function = simplified_func

            simplified_func = function._simplify2(var_dict)
            is_simpler |= (simplified_func is not None)
            function = simplified_func or function

        if is_simpler:
            return SingleRegion(condition, function)
        return None


class ConditionalFunction(MapElement):
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
        for condition, _ in regions:
            assert condition.has_promise(IsCondition)
        self.regions = []
        for region in regions:
            if isinstance(region, SingleRegion):
                self.regions.append(region)
            else:
                self.regions.append(SingleRegion(region[0], convert_to_map(region[1])))
        variables = sum([region.vars for region in self.regions], [])

        super().__init__(list(set(variables)))

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

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        if len(var_dict) == 0 and len(func_dict) == 0:
            return self

        regions = [region._call_with_dict(var_dict, func_dict) for region in self.regions]
        return ConditionalFunction(regions)

    # <editor-fold desc=" ------------------------ Simplifiers and Validators ------------------------ ">

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
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

            simplified_region = region._simplify2(var_dict)
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
                    condition_union = condition_union.simplify2()
                    regions[i] = SingleRegion(condition_union, comb_elem)
                    break
            else:
                regions.append(region)

        # TODO: The conditions in a conditional function should cover the whole space, so a single region
        #       must always have a TrueCondition. However, it is not always true that it is easy to check
        #       that the condition is true. Should I keep this check here or not?
        if len(regions) == 1:  # and regions[0][0] is TrueCondition:
            return regions[0].function

        return ConditionalFunction(regions) if is_simpler else None

    def mult_condition_by_element(var_dict: VarDict) -> Optional[MapElement]:
        a, b = [var_dict.get(v, v) for v in var_dict]
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
    def new_assignment_simplify(ranged_cond: MapElement, var_dict: VarDict) -> Optional[MapElement]:
        assert isinstance(ranged_cond, RangeCondition)
        cond_function = ranged_cond.function
        if not isinstance(cond_function, ConditionalFunction):
            return None

        f_range = ranged_cond.range
        return UnionCondition([condition & RangeCondition(func, f_range) for condition, func in cond_function.regions])

    @staticmethod
    def _bool_var_simplifier(
        map_elem: MapElement, var_dict: VarDict
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

        if not (isinstance(cond1, RangeCondition) and isinstance(cond2, RangeCondition)):
            return None
        assign1 = cond1.as_assignment()
        assign2 = cond2.as_assignment()
        if assign1 is None or assign2 is None:
            return None

        v1 = assign1[0]
        v2 = assign2[0]
        if not (isinstance(v1, BoolVar) and v1 is v2):
            return None

        assigned_value1 = assign1[1]
        assigned_value2 = assign2[1]

        if (assigned_value1, assigned_value2) == (0, 1):
            return (value1 + (value2 - value1) * v1).simplify2()

        if (assigned_value1, assigned_value2) == (1, 0):
            return (value2 + (value1 - value2) * v1).simplify2()

        raise Exception(
            f"The assigned values should be 0 and 1, but instead got {assigned_value1} and {assigned_value2}"
        )

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


Mult.register_simplifier(ConditionalFunction.mult_condition_by_element)
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