import operator
from typing import List, Tuple, Optional

from mapping_field.mapping_field import MapElement, ExtElement, MapElementConstant, VarDict, FuncDict, params_to_maps
from mapping_field.conditions import TrueCondition, Condition, FalseCondition, MapElementProcessor, _ListCondition
from mapping_field.ranged_condition import SingleAssignmentCondition
from mapping_field.serializable import DefaultSerializable


class ConditionalFunction(MapElement, DefaultSerializable):
    """
    A conditional function of the form:
       1_(cond_1) * f_1 + 1_(cond_2) * f_2 + ... + 1_(cond_n) * f_n

    Working under the assumption that the conditions do not intersect, and cover the whole space, namely
    the form a decomposition of the whole space.
    """

    @staticmethod
    def always(map: MapElement):
        return ConditionalFunction([(TrueCondition, map)])

    def __init__(self, regions: List[Tuple[Condition, MapElement]]):
        self.regions = regions
        variables = sum([region[0].vars + region[1].vars for region in regions], [])

        super().__init__(list(set(variables)))

    def to_string(self, vars_str_list: List[str]):
        # TODO: fix this printing function
        return ' , '.join([f'( {repr(condition)} -> {repr(map)} )' for (condition, map) in self.regions])

    def pretty_str(self):
        return '\n  @@       +      @@  \n'.join([
            f'Given:  \n{condition.pretty_str() if isinstance(condition, _ListCondition) else repr(condition)} \n -->  {repr(map)}'
            for condition, map in self.regions
        ])

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
        regions: List[Tuple[Condition, MapElement]] = []
        for (cond1, elem1) in self.regions:
            for (cond2, elem2) in other.regions:
                cond_prod = (cond1 * cond2).simplify()
                if cond_prod != FalseCondition:
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

    def rdiv(self, other) -> 'ConditionalFunction':
        return ConditionalFunction.always(other)._op(self, operator.truediv)

    # </editor-fold>

    # Override when needed
    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        if len(var_dict) == 0 and len(func_dict) == 0:
            return self

        # TODO: call with dict in the condition also
        regions = [(region[0], region[1]._call_with_dict(var_dict, func_dict))
                   for region in self.regions]
        return ConditionalFunction(regions)

    def _simplify2(self) -> Optional['MapElement']:
        return self._simplify_with_var_values2({v: v for v in self.vars})

    def _simplify_with_var_values2(self, var_dict: VarDict) -> 'MapElement':
        regions = []
        for condition, func in self.regions:
            condition = condition.simplify()
            if condition == FalseCondition:
                continue
            func = func._simplify_with_var_values2(var_dict) or func
            if isinstance(condition, MapElementProcessor):
                func = condition.process_function(func)
                func = func._simplify_with_var_values2(var_dict) or func

            for i, (prev_cond, prev_func) in enumerate(regions):
                if prev_func == func:
                    condition_union = prev_cond | condition
                    condition_union = condition_union.simplify()
                    regions[i][0] = condition_union
                    break
            else:
                regions.append([condition, func])

        if len(regions) == 1 and regions[0][0] is TrueCondition:
            return regions[0][1]

        # If there is a region (assignment -> func1) and another region (cond -> func2), such that
        # func2(assignment) = func, then we can combine them.

        assignment_regions = []
        other_regions = []
        for condition, func in regions:
            var_dict = SingleAssignmentCondition.as_assignment_dict(condition)
            if var_dict is not None:
                assignment_regions.append((var_dict, condition, func))
            else:
                other_regions.append((condition, func))

        for var_dict, condition, func in assignment_regions:
            for other_region in other_regions:
                other_cond, other_func = other_region
                if other_func(var_dict) == func:
                    other_regions = [region for region in other_regions if (region is not other_region)]
                    union_condition = (other_cond | condition).simplify()
                    other_regions.append((union_condition, other_func))
                    break
            else:
                other_regions.append((condition, func))

        return ConditionalFunction([tuple(region) for region in other_regions])


def ReLU(map_elem: MapElement):
    zero = MapElementConstant.zero
    if isinstance(map_elem, ConditionalFunction):
        regions = []
        for condition, func in map_elem.regions:
            non_negative = (func >= 0)
            if non_negative == TrueCondition:
                # Make your and my life a little bit simpler
                regions.append( (condition, func) )
            elif non_negative == FalseCondition:
                regions.append( (condition, zero) )
            else:
                regions.append( (condition * non_negative, func) )
                regions.append( (condition * (func < 0), zero) )
        regions = [(cond, func) for cond, func in regions if FalseCondition != cond]
        return ConditionalFunction(regions)
    return ConditionalFunction([
        ((map_elem >= 0), map_elem),
        ((map_elem < 0), zero)
    ])