from abc import abstractmethod
from typing import List, Tuple, Optional
import operator

from mapping_field import MapElement, Var, VarDict, FuncDict, MapElementConstant

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

    @classmethod
    def intersection(cls, condition1: 'Condition', condition2: 'Condition') -> Tuple['Condition', bool]:
        """
        Tries to intersect the conditions in the argument, and returns the result.
        The second value returned by this function is True if the intersection is "simpler" than just

            ConditionIntersection([self, condition])

        For example:
            intersection( 0<x<10   , 5<x<15 ) = (5<x<10), True
            intersection( 0<x+y<10 , x=5    ) = ConditionIntersection((-5<y<5), ( x=5 ) ), True
            intersection( 0<x<10   , 5<y<15 ) = ConditionIntersection((0<x<10), (5<y<15)), False
        """
        condition, is_simpler = condition1 & condition2
        if is_simpler:
            return condition, is_simpler
        return condition2 & condition1

    def __and__(self, condition: 'Condition') -> Tuple['Condition', bool]:
        """
        Tries to intersect this condition with the condition in the argument, and returns the result.
        The second value returned by this function is True if the intersection is "simpler" than just
            ConditionIntersection([self, condition])
        Similar to Condition.intersection(...).
        Not that depending on the condition type, sometimes cond1 & cond2 can be simpler than just the intersection
        while cond2 & cond1 not. Hence, if you really want to check if it can be simpler, check both, or use
        Condition.intersection(...) directly.
        """
        if isinstance(condition, BinaryCondition) or isinstance(condition, ConditionIntersection):
            return condition & self

        if self == condition:
            return self, True

        return ConditionIntersection([self, condition]), False

    def __mul__(self, condition: 'Condition'):
        """
        Same as __and__, but does not return the second bool argument
        """
        return (self & condition)[0]

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
        if isinstance(other, bool):
            return other == self.value
        return False

    def __and__(self, condition: Condition) -> Tuple[Condition, bool]:
        if self.value:
            return condition, True
        return self, True

TrueCondition  = BinaryCondition(True)
FalseCondition = BinaryCondition(False)


class ListCondition(Condition):

    def __init_subclass__(cls, op_type, op_dunder_name: str, join_delim: str,
                          zero_condition: Condition, one_condition: Condition):
        cls.op_type = op_type
        cls.join_delim = join_delim
        cls.zero_condition = zero_condition
        cls.one_condition = one_condition
        setattr(cls, op_dunder_name, cls.op)

    def __init__(self, conditions: List[Condition], simplified: bool = False):
        super().__init__(
            list(set(sum([condition.vars for condition in conditions],[])))
        )
        self.conditions = conditions
        self._simplified = simplified

    def __repr__(self):
        conditions_rep = self.__class__.join_delim.join(repr(condition) for condition in self.conditions)
        return f'[{conditions_rep}]'

    def op(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, BinaryCondition):
            return self.__class__.op_type(condition , self)

        if isinstance(condition, self.__class__):
            return self.__class__([*self.conditions, *condition.conditions]), False

        return self.__class__([*self.conditions, condition]), False


    def simplify(self):
        if self._simplified:
            return self

        cls = self.__class__

        final_conditions = []
        conditions = self.conditions

        for condition in conditions:

            condition = condition.simplify()

            if condition == cls.one_condition:
                continue

            if isinstance(condition, cls):
                conditions.extend(condition.conditions)
                continue

            while (True):
                # Check if this new condition intersects in a special way with an existing condition.
                # Each time this loop repeats itself, the conditions array's size must decrease by 1, so it cannot
                # continue forever.
                if condition == cls.zero_condition:
                    return cls.zero_condition

                for existing_condition in final_conditions:
                    prod_cond, is_simpler = self.__class__.op_type(existing_condition, condition)
                    if is_simpler:
                        final_conditions = [cond for cond in final_conditions if cond != existing_condition]
                        condition = prod_cond.simplify()
                        break

                    prod_cond, is_simpler = self.__class__.op_type(condition, existing_condition)
                    if is_simpler:
                        final_conditions = [cond for cond in final_conditions if cond != existing_condition]
                        condition = prod_cond.simplify()
                        break

                else:
                    # new condition cannot be intersected in a special way
                    final_conditions.append(condition)
                    break

        if len(final_conditions) == 0:
            return self.__class__.one_condition

        if len(final_conditions) == 1:
            return final_conditions[0]

        return cls(final_conditions, simplified = True)

    def _eq_simplified(self, other: Condition):
        if not isinstance(other, self.__class__):
            return False
        if len(self.conditions) != len(other.conditions):
            return False

        for condition in self.conditions:
            if condition not in other.conditions:
                return False

        return True

class ConditionIntersection(ListCondition,
                            op_type=operator.and_, op_dunder_name='__and__', join_delim=' & ',
                            zero_condition = FalseCondition, one_condition = TrueCondition):

    def __init__(self, conditions: List[Condition], simplified: bool = False):
        super().__init__(conditions, simplified)


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

    def __and__(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, AssignmentCondition):
            # Combine the dictionaries, if there are no different assignments for the same variables
            for key, value in self.var_dict.items():
                if condition.var_dict.get(key, value) != value:
                    return FalseCondition, True
            return AssignmentCondition({**self.var_dict, **condition.var_dict}), True

        return super().__and__(condition)


class RangeCondition(Condition):

    def __init__(self, function: MapElement, f_range: Range, simplified: bool = False):
        super().__init__(function.vars)
        self.function = function
        self.range = f_range
        self._simplified = simplified

    def __repr__(self):
        lower = '' if self.range[0] == float('-inf') else f'{self.range[0]} <= '
        upper = '' if self.range[1] == float('inf') else f' < {self.range[1]}'
        return f'{lower}{repr(self.function)}{upper}'

        return f'{self.range[0]} <= {repr(self.function)} < {self.range[1]}'

    def _eq_simplified(self, condition: 'Condition') -> bool:
        if isinstance(condition, RangeCondition):
            return self.function == condition.function and self.range == condition.range
        return super()._eq_simplified(condition)

    def __and__(self, condition: Condition) -> Tuple[Condition, bool]:
        if isinstance(condition, RangeCondition):
            if self.function == condition.function:
                low = max(self.range[0], condition.range[0])
                high = min(self.range[1], condition.range[1])
                if high <= low:
                    return FalseCondition, True
                return RangeCondition(self.function, (low, high)), True

        if isinstance(condition, AssignmentCondition):
            function = self.function(condition.var_dict)
            if function != self.function:
                # TODO: beware of infinite loops...
                return ConditionIntersection([RangeCondition(function, self.range), condition]), True

        return super().__and__(condition)


    def simplify(self) -> 'Condition':
        if self.range[0] >= self.range[1]:
            return FalseCondition

        condition = self
        while (isinstance(condition, RangeCondition) and (not condition._simplified) and
               isinstance(condition.function, RangeTransformer)):
            new_condition = condition.function.transform_range(condition.range)
            if new_condition is None:
                return condition
            condition = new_condition

        return condition


class RangeTransformer:

    @abstractmethod
    def transform_range(self, range_values: Range) -> Optional[Condition]:
        """
        Try to simplify being in the given range (should be called on a MapElement).
        If cannot be simplified, return None.
        """
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

    def to_string(self, vars_str_list: List[str]):
        # TODO: fix this printing function
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
                cond_prod = (cond1 * cond2).simplify()
                if cond_prod != FalseCondition:
                    regions.append(( cond_prod, op_func(elem1, elem2)))
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
        regions = [(region[0].simplify(), region[1].simplify())
                   for region in self.regions]
        return ConditionalFunction(regions)

def ReLU(map_elem: MapElement):
    return ConditionalFunction([
        (RangeCondition(map_elem, (0, float('inf'))), map_elem),
        (RangeCondition(map_elem, (float('-inf'), 0)), MapElementConstant(0))
    ])