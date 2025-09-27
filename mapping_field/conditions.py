from abc import abstractmethod
from typing import List, Optional, Dict, Tuple, Union
import operator

from mapping_field import MapElement, Var, VarDict, FuncDict, MapElementFromFunction, MapElementConstant
from mapping_field.arithmetics import _Add

Range = Tuple[float, float]

class Condition:

    """
    A 0\1 function.
    Use the product between conditions to take the intersection
    # TODO: change into a MapElement?
    """

    def __init__(self, variables: List['Var']):
        """
        The 'variables' are the ordered list used when calling the function, as in f(a_1,...,a_n).
        """
        if len(variables) > len(set(variables)):
            raise Exception(f'Function must have distinct variables')
        self.vars = variables
        self.num_vars = len(variables)

    def __eq__(self, other: 'Condition'):
        '''
        Checks if both conditions are "possibly" equal.
        If returns True, then they are equal.
        If returns False, they might still be equal, but could not find the right way to compare them.
        '''
        cond1 = self.simplify()
        cond2 = other.simplify()
        return cond1._eq_simplified(cond2)

    # override if needed
    def _eq_simplified(self, other: 'Condition') -> bool:
        '''
        Checks if this condition as the same as other, assuming both are simplified.
        See remark in __eq__ above.
        '''
        return super().__eq__(other)

    def __mul__(self, condition: 'Condition'):
        if isinstance(condition, BinaryCondition) or isinstance(condition, ConditionIntersection):
            return condition * self
        else:
            return ConditionIntersection([self, condition])

    def simplify(self) -> 'Condition':
        return self


class BinaryCondition(Condition):
    """
    An always True \ False condition.
    """

    def __init__(self, value: bool):
        super().__init__(variables=[])
        self.value = value

    def __repr__(self):
        return repr(self.value)

    def _eq_simplified(self, other: 'Condition') -> bool:
        if isinstance(other, BinaryCondition):
            return other.value == self.value
        return False

    def __mul__(self, condition: 'Condition'):
        if self.value:
            return condition
        return self

TrueCondition = BinaryCondition(True)
FalseCondition = BinaryCondition(False)


class ConditionIntersection(Condition):

    def __init__(self, conditions: List[Condition], simplified: bool = False):
        super().__init__(
            list(set(sum([condition.vars for condition in conditions],[])))
        )
        self.conditions = conditions
        self._simplified = simplified

    def __repr__(self):
        conditions_rep = ', '.join(repr(condition) for condition in self.conditions)
        return f'[{conditions_rep}]'

    def __mul__(self, condition: 'Condition') -> Condition:
        if isinstance(condition, BinaryCondition):
            return condition * self

        return ConditionIntersection([*self.conditions, condition])

    def simplify(self):
        if self._simplified:
            return self

        if FalseCondition in self.conditions:
            return FalseCondition

        conditions = []
        for condition in self.conditions:
            condition = condition.simplify()
            if condition == TrueCondition:
                continue
            if condition not in conditions:
                conditions.append(condition)

        if len(conditions) == 0:
            return TrueCondition

        if len(conditions) == 1:
            return conditions[0]

        if len(conditions) == len(self.conditions):
            self._simplified = True
            return self

        return ConditionIntersection(conditions, simplified = True)

    def _eq_simplified(self, other: Condition):
        if not isinstance(other, ConditionIntersection):
            return False
        if len(self.conditions) != len(other.conditions):
            return False

        for condition in self.conditions:
            if condition not in other.conditions:
                return False

        return True


class AssignmentCondition(Condition):

    def __init__(self, var_dict: VarDict):
        super().__init__(list(var_dict.keys()))
        self.var_dict = var_dict

    def __repr__(self):
        return repr(self.var_dict)

    def _eq_simplified(self, condition: 'Condition') -> bool:
        if isinstance(condition, AssignmentCondition):
            return self.var_dict == condition.var_dict
        return super()._eq_simplified(condition)

    def __mul__(self, condition: 'Condition'):
        if isinstance(condition, AssignmentCondition):
            # Combine the dictionaries, if there are no different assignments for the same variables
            for key, value in self.var_dict.items():
                if condition.var_dict.get(key, value) != value:
                    return FalseCondition
            return AssignmentCondition({**self.var_dict, **condition.var_dict})

        return super().__mul__(condition)


class RangeCondition(Condition):

    def __init__(self, function: MapElement, f_range: Range):
        super().__init__(function.vars)
        self.function = function
        self.range = f_range

    def __repr__(self):
        return f'{self.range[0]} <= {repr(self.function)} < {self.range[1]}'

    def _eq_simplified(self, condition: 'Condition') -> bool:
        if isinstance(condition, RangeCondition):
            return self.function == condition.function and self.range == condition.range
        return super()._eq_simplified(condition)

    def __mul__(self, condition: 'Condition'):
        if isinstance(condition, RangeCondition):
            if self.function == condition.function:
                low = max(self.range[0], condition.range[0])
                high = min(self.range[1], condition.range[1])
                if high <= low:
                    return FalseCondition
                return RangeCondition(self.function, (low, high))
        return super().__mul__(condition)

    def simplify(self) -> 'Condition':
        if self.range[0] >= self.range[1]:
            return FalseCondition

        condition = self
        while isinstance(condition, RangeCondition) and  isinstance(condition.function, RangeTransformer):
            condition = condition.function.transform_range(condition.range)

        return condition


class RangeTransformer:

    @abstractmethod
    def transform_range(self, range_values: Range) -> Union[RangeCondition , AssignmentCondition]:
        pass

# ========================================================================= #

class ConditionalFunction(MapElement):
    """
    A conditional function of the form:
       1_(cond_1) * f_1 + 1_(cond_2) * f_2 + ... + 1_(cond_n) * f_n

    Working under the assumption that the conditions do not intersect
    """

    @staticmethod
    def always(map: MapElement):
        return ConditionalFunction([(TrueCondition, map)])

    def __init__(self, regions: List[Tuple[Condition, MapElement]]):
        self.regions = regions
        variables = sum([region[0].vars + region[1].vars for region in regions], [])

        super().__init__(list(set(variables)))

    def __repr__(self):
        return ' , '.join([f'( {repr(condition)} -> {repr(map)} )' for (condition, map) in self.regions])

    def __eq__(self, other: 'Condition') -> bool:
        if not isinstance(other, ConditionalFunction):
            return super().__eq__(other)

        if len(self.regions) != len(other.regions):
            return False

        for region in self.regions:
            if region not in other.regions:
                return False

        return True

    # <editor-fold desc=" ------------------------ arithmetics ------------------------">

    def _op(self, other: 'MapElement', op_func):
        if not isinstance(other, ConditionalFunction):
            other = ConditionalFunction.always(other)
        regions: List[Tuple[Condition, MapElement]] = []
        for (cond1, elem1) in self.regions:
            for (cond2, elem2) in other.regions:
                cond_prod = cond1 * cond2
                if cond_prod != FalseCondition:
                    regions.append(( cond1 * cond2, op_func(elem1, elem2)))
        return ConditionalFunction(regions)

    def __add__(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.add)

    def __radd__(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.add)

    def __mul__(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.mul)

    def __rmul__(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.mul)

    def __sub__(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.sub)

    def __rsub__(self, other: MapElement) -> 'ConditionalFunction':
        return ConditionalFunction.always(other)._op(self, operator.sub)

    def __truediv__(self, other: MapElement) -> 'ConditionalFunction':
        return self._op(other, operator.truediv)

    def __rtruediv__(self, other) -> 'ConditionalFunction':
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

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        return self # TODO: implement

def ReLU(map_elem: MapElement):
    return ConditionalFunction([
        (RangeCondition(map_elem, (0, float('inf'))), map_elem),
        (RangeCondition(map_elem, (float('-inf'), 0)), MapElementConstant(0))
    ])