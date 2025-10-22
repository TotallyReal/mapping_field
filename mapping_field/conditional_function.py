import operator
from typing import List, Tuple, Optional

from mapping_field.binary_expansion import BoolVar
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
        inner_str = ' ; '.join([f' {repr(condition)} -> {repr(map)} ' for (condition, map) in self.regions])
        return f'[{inner_str}]'

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

    def _simplify_with_var_values2(self, var_dict: VarDict) -> 'MapElement':

        def combinable(condition1: Condition, elem1: MapElement, condition2: Condition, elem2:MapElement) \
                -> Optional[MapElement]:
            if elem1 == elem2:
                return elem1

            var_dict = SingleAssignmentCondition.as_assignment_dict(condition1)
            if var_dict is not None and elem2(var_dict) == elem1:
                return elem2

            var_dict = SingleAssignmentCondition.as_assignment_dict(condition2)
            if var_dict is not None and elem1(var_dict) == elem2:
                return elem1

            return None

        regions = []
        is_simpler = False
        for condition, func in self.regions:
            # TODO: use var_dict to simplify the conditions
            simplified_condition = condition.simplify()
            # TODO: Use a simplifier process like in map elements instead
            is_simpler |= (simplified_condition is not condition)
            condition = simplified_condition

            if condition == FalseCondition:
                continue
            simplified_func = func._simplify2(var_dict)
            is_simpler |= (simplified_func is not None)
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
                    condition_union = condition_union.simplify()
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


def bool_var_simplifier(map_elem: MapElement, var_dict: VarDict) -> Optional[MapElement]:
    assert isinstance(map_elem, ConditionalFunction)

    if len(map_elem.regions) != 2:
        return None

    cond1, func1 = map_elem.regions[0]
    cond2, func2 = map_elem.regions[1]
    value1 = func1.evaluate()
    value2 = func2.evaluate()
    if value1 is None or value2 is None:
        return None

    if not (isinstance(cond1, SingleAssignmentCondition) and isinstance(cond2, SingleAssignmentCondition)):
        return None

    v1 = cond1.var
    v2 = cond2.var
    if not (isinstance(v1, BoolVar) and v1 is v2):
        return None

    assigned_value1 = cond1.value
    assigned_value2 = cond2.value

    if (assigned_value1, assigned_value2) == (0, 1):
        return (value1 + (value2 - value1) * v1).simplify2()

    if (assigned_value1, assigned_value2) == (1, 0):
        return (value2 + (value1 - value2) * v1).simplify2()

    raise Exception(f'The assigned values should be 0 and 1, but instead got {assigned_value1} and {assigned_value2}')

ConditionalFunction.register_class_simplifier(bool_var_simplifier)


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
                regions.append( (condition * (func > 0), func) )
                regions.append( (condition * (func <= 0), zero) )
        regions = [(cond, func) for cond, func in regions if FalseCondition != cond]
        return ConditionalFunction(regions)
    return ConditionalFunction([
        ((map_elem > 0), map_elem),
        ((map_elem <= 0), zero)
    ])