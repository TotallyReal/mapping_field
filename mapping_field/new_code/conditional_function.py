import operator

from typing import Dict, List, Optional, Tuple

from mapping_field.field import ExtElement
from mapping_field.new_code.arithmetics import Mult
from mapping_field.new_code.conditions import FalseCondition, TrueCondition, UnionCondition
from mapping_field.new_code.mapping_field import (
    FuncDict, MapElement, MapElementConstant, MapElementProcessor, OutputValidator, Var, VarDict,
    convert_to_map, params_to_maps,
)
from mapping_field.new_code.promises import IsCondition, IsIntegral
from mapping_field.new_code.ranged_condition import RangeCondition


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

    def __init__(self, regions: List[Tuple[MapElement, MapElement]]):
        for condition, _ in regions:
            assert condition.has_promise(IsCondition)
        self.regions = [(condition, convert_to_map(func)) for condition, func in regions]
        variables = sum([region[0].vars + region[1].vars for region in self.regions], [])

        super().__init__(list(set(variables)))

    def to_string(self, vars_to_str: Dict[Var, str]):
        # TODO: fix this printing function
        inner_str = ' ; '.join([f' {repr(condition)} -> {repr(map)} ' for (condition, map) in self.regions])
        return f'[{inner_str}]'
#
#     def pretty_str(self):
#         return '\n  @@       +      @@  \n'.join([
#             f'Given:  \n{condition.pretty_str() if isinstance(condition, _ListCondition) else repr(condition)} \n -->  {repr(map)}'
#             for condition, map in self.regions
#         ])

    def evaluate(self) -> Optional[ExtElement]:
        values = [func.evaluate() for _, func in self.regions]
        if len(values) == 0:
            raise Exception('Conditional Map should not be empty')
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

    def _op(self, other: MapElement, op_func) -> 'ConditionalFunction':
        if isinstance(other, int):
            other = MapElementConstant(other)
        if not isinstance(other, ConditionalFunction):
            other = ConditionalFunction.always(other)
        regions: List[Tuple[MapElement, MapElement]] = []
        for (cond1, elem1) in self.regions:
            for (cond2, elem2) in other.regions:
                cond_prod = (cond1 & cond2).simplify2()
                if cond_prod is not FalseCondition:
                    regions.append((cond_prod, op_func(elem1, elem2)))
        return ConditionalFunction(regions).simplify2()

    def add(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.add)

    def radd(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.add)

    def mul(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.mul)

    def rmul(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.mul)

    def sub(self, other: MapElement) -> 'ConditionalFunction':
        if other == 0:
            return self
        return self._op(other, operator.sub)

    def rsub(self, other: MapElement) -> 'ConditionalFunction':
        return ConditionalFunction.always(other)._op(self, operator.sub)

    def div(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.truediv)

    def rdiv(self, other: MapElement) -> 'ConditionalFunction':
        return ConditionalFunction.always(other)._op(self, operator.truediv)

    # </editor-fold>

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        if len(var_dict) == 0 and len(func_dict) == 0:
            return self

        # TODO: call with dict in the condition also
        regions = [(region[0]._call_with_dict(var_dict, func_dict), region[1]._call_with_dict(var_dict, func_dict))
                   for region in self.regions]
        return ConditionalFunction(regions)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> 'MapElement':
        # TODO: combine this simplification with the _ListCondition simplification for a general simplification of
        #       a commutative associative binary function.

        def combinable(condition1: MapElement, elem1: MapElement, condition2: MapElement, elem2:MapElement) \
                -> Optional[MapElement]:
            """
            Look for a single MapFunctions elem such that :
                elem(condition1) = elem1 ,
                elem(condition2) = elem2
            """
            if elem1 == elem2:
                return elem1

            if isinstance(condition1, MapElementProcessor):
                # TODO: use __call__(condition1) instead?
                processed_elem2 = condition1.process_function(elem2)
                if processed_elem2 == elem1:
                    return elem2

            if isinstance(condition2, MapElementProcessor):
                # TODO: use __call__(condition1) instead?
                processed_elem1 = condition2.process_function(elem1)
                if processed_elem1 == elem2:
                    return elem1

            return None

        regions = []
        is_simpler = False
        for condition, func in self.regions:

            # TODO: use var_dict to simplify the conditions
            simplified_condition = condition._simplify2()
            if simplified_condition is not None:
                is_simpler = True
            condition = simplified_condition or condition

            if condition == FalseCondition:
                continue
            simplified_func = func._simplify2(var_dict)
            if simplified_func is not None:
                is_simpler = True
            func = simplified_func or func

            if isinstance(condition, MapElementProcessor):
                # TODO: Use a process_function process like in map elements instead
                simplified_func = condition.process_function(func)
                is_simpler |= (simplified_func is not func)
                func = simplified_func

                simplified_func = func._simplify2(var_dict)
                is_simpler |= (simplified_func is not None)
                func = simplified_func or func

            for i, (prev_cond, prev_func) in enumerate(regions):
                comb_elem = combinable(prev_cond, prev_func, condition, func)
                if comb_elem is not None:
                    is_simpler = True
                    condition_union = prev_cond | condition
                    condition_union = condition_union.simplify2()
                    regions[i] = [condition_union, comb_elem]
                    break
            else:
                regions.append([condition, func])

        # TODO: The conditions in a conditional function should cover the whole space, so a single region
        #       must always have a TrueCondition. However, it is not always true that it is easy to check
        #       that the condition is true. Should I keep this check here or not?
        if len(regions) == 1: # and regions[0][0] is TrueCondition:
            return regions[0][1]

        return ConditionalFunction([tuple(region) for region in regions]) if is_simpler else None

def promise_validate_conditional_function(validator: OutputValidator, elem: MapElement) -> bool:
    if not isinstance(elem, ConditionalFunction):
        return False
    for condition, function in elem.regions:
        if not function.has_promise(validator):
            return False
    return True
# TODO: Make it work for any validator in the future
IsIntegral.register_validator(lambda elem: promise_validate_conditional_function(IsIntegral, elem))
# TODO: Don't think of ConditionalFunction as a condition for now
# IsCondition.register_validator(lambda elem: promise_validate_conditional_function(IsCondition, elem))

def mult_condition_by_element(var_dict: VarDict) -> Optional[MapElement]:
    a, b = [var_dict.get(v,v) for v in var_dict]
    a_is_cond = a.has_promise(IsCondition)
    b_is_cond = b.has_promise(IsCondition)
    if a_is_cond == b_is_cond:
        return None

    if b_is_cond:
        a, b = b, a
    if isinstance(a, (Var, MapElementConstant)):
        return None
    return ConditionalFunction([(a, b), (~a, MapElementConstant.zero)])

Mult.register_simplifier(mult_condition_by_element)

#
# _T = TypeVar('_T', bound=Union[Condition, MapElement])
#
# # TODO: Think of a better way of overriding this function
# #       Also, should __mul__ be used for condition * condition = condition also, or only use __and__ for that?
# @params_to_maps
# def condition_mul(self, func: MapElement) -> MapElement:
#
#     return ConditionalFunction([
#         (self, func),
#         (~self, MapElementConstant.zero)
#     ]).simplify2()
# Condition.__mul__ = condition_mul
# Condition.__rmul__ = condition_mul
#
#
# def bool_var_simplifier(map_elem: MapElement, var_dict: VarDict) -> Optional[MapElement]:
#     assert isinstance(map_elem, ConditionalFunction)
#
#     if len(map_elem.regions) != 2:
#         return None
#
#     cond1, func1 = map_elem.regions[0]
#     cond2, func2 = map_elem.regions[1]
#     value1 = func1.evaluate()
#     value2 = func2.evaluate()
#     if value1 is None or value2 is None:
#         return None
#
#     if not (isinstance(cond1, SingleAssignmentCondition) and isinstance(cond2, SingleAssignmentCondition)):
#         return None
#
#     v1 = cond1.var
#     v2 = cond2.var
#     if not (isinstance(v1, BoolVar) and v1 is v2):
#         return None
#
#     assigned_value1 = cond1.value
#     assigned_value2 = cond2.value
#
#     if (assigned_value1, assigned_value2) == (0, 1):
#         return (value1 + (value2 - value1) * v1).simplify2()
#
#     if (assigned_value1, assigned_value2) == (1, 0):
#         return (value2 + (value1 - value2) * v1).simplify2()
#
#     raise Exception(f'The assigned values should be 0 and 1, but instead got {assigned_value1} and {assigned_value2}')
#
# ConditionalFunction.register_class_simplifier(bool_var_simplifier)
#
# # TODO: another refactoring nightmare
# original_assignment_simplify = GeneralAssignment.simplify
#
def new_assignment_simplify(ranged_cond:MapElement, var_dict:VarDict) -> Optional[MapElement]:
    assert isinstance(ranged_cond, RangeCondition)
    cond_function = ranged_cond.function
    if not isinstance(cond_function, ConditionalFunction):
        return None

    f_range = ranged_cond.range
    return UnionCondition([condition & RangeCondition(func, f_range) for condition, func in cond_function.regions])
RangeCondition.register_class_simplifier(new_assignment_simplify)
#
def ReLU(map_elem: MapElement):
    zero = MapElementConstant.zero
    if isinstance(map_elem, ConditionalFunction):
        regions = []
        for condition, func in map_elem.regions:
            non_negative = (func >= 0)
            if (func >= 0) == TrueCondition:
                # Make your and my life a little bit simpler
                regions.append( (condition, func) )
            elif (func <= 0) == FalseCondition:
                regions.append( (condition, zero) )
            else:
                regions.append( (condition & (func > 0), func) )
                regions.append( (condition & (func <= 0), zero) )
        regions = [(cond, func) for cond, func in regions if FalseCondition != cond]
        return ConditionalFunction(regions)
    return ConditionalFunction([
        ((map_elem > 0), map_elem),
        ((map_elem <= 0), zero)
    ])