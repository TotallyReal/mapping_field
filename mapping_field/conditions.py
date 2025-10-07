from abc import abstractmethod
from typing import List, Tuple
import operator

from mapping_field import MapElement, Var, VarDict, FuncDict, ExtElement, MapElementConstant

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
        if not isinstance(condition, Condition):
            return NotImplemented

        if isinstance(condition, BinaryCondition) or isinstance(condition, ConditionIntersection):
            return condition & self

        if self == condition:
            return self, True

        return ConditionIntersection([self, condition]), False

    @classmethod
    def union(cls, condition1: 'Condition', condition2: 'Condition') -> Tuple['Condition', bool]:
        """
        Similar to Condition.intersection(...), but with union.
        """
        condition, is_simpler = condition1 | condition2
        if is_simpler:
            return condition, is_simpler
        return condition2 | condition1

    def __or__(self, condition: 'Condition') -> Tuple['Condition', bool]:
        """
        Similar to __and__, but for __or__
        """
        if not isinstance(condition, Condition):
            return NotImplemented

        if isinstance(condition, BinaryCondition) or isinstance(condition, ConditionUnion):
            return condition | self

        if self == condition:
            return self, True

        return ConditionUnion([self, condition]), False

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
        return (condition if self.value else FalseCondition), True

    def __or__(self, condition: 'Condition') -> Tuple['Condition', bool]:
        return (condition if not self.value else TrueCondition), True

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
        conditions = self.conditions.copy()

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
                    prod_cond, is_simpler = cls.op_type(existing_condition, condition)
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

class ConditionUnion(ListCondition,
                            op_type = operator.or_, op_dunder_name = '__or__', join_delim = ' | ',
                            zero_condition = TrueCondition, one_condition = FalseCondition):

    def __init__(self, conditions: List[Condition], simplified: bool = False):
        super().__init__(conditions, simplified)

    def __and__(self, condition: Condition) -> Tuple[Condition, bool]:
        # We have two types of simplifications:
        # 1. one condition is contained in the other
        # 2. Both conditions are unions, of the form:
        #       A_1 | A_2 | ... | A_n   ,   B_1 | A_2 | ... | A_n
        #    and A_1 * B_1 can be simplified, in which case we return
        #       (A_1 * B_1) | A_2 | A_3 | ... | A_n

        if isinstance(condition, BinaryCondition):
            return condition & self

        conditions1 = self.conditions.copy()
        conditions2 = condition.conditions.copy() if isinstance(condition, ConditionUnion) else [condition]

        swapped = False
        if len(conditions1) < len(conditions2):
            conditions1, conditions2 = conditions2, conditions1
            swapped = True

        n1 = len(conditions1)
        used_positions = [False] * n1
        special_condition = None

        for cond2 in conditions2:
            for i in range(n1):
                if not used_positions[i] and conditions1[i] == cond2:
                    used_positions[i] = True
                    break
            else:
                if special_condition is not None:
                    return super().__and__(condition)
                special_condition = cond2

        if special_condition is None:
            # Full containement
            return (self if swapped else condition), True

        if len(conditions1) == len(conditions2):
            second_special_condition = None
            for i in range(n1):
                if not used_positions[i]:
                    second_special_condition = conditions1[i]
                    break

            prod, is_simpler = special_condition & second_special_condition
            if is_simpler:
                conditions = [c for flag, c in zip(used_positions, conditions1) if flag]
                return ConditionUnion(conditions + [prod]), True

        return super().__and__(condition)




# ========================================================================= #

class MapElementProcessor:

    @abstractmethod
    def process(self, func: MapElement) -> MapElement:
        pass

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

    def __init__(self, regions: List[Tuple[Condition, MapElement]]):
        self.regions = regions
        variables = sum([region[0].vars + region[1].vars for region in regions], [])

        super().__init__(list(set(variables)))

    def to_string(self, vars_str_list: List[str]):
        # TODO: fix this printing function
        return ' , '.join([f'( {repr(condition)} -> {repr(map)} )' for (condition, map) in self.regions])

    def evaluate(self) -> ExtElement:
        condition, func = self.regions[0]
        value = func.evaluate()
        assert all([value == func.evaluate() for _, func in self.regions])
        return value

    def __eq__(self, other: MapElement) -> bool:
        if (isinstance(other, MapElement) or isinstance(other, int)):
            return (self-other).simplify().is_zero()
        return super().__eq__(other)

    # <editor-fold desc=" ------------------------ arithmetics ------------------------">

    def _op(self, other: 'MapElement', op_func) -> 'ConditionalFunction':
        if isinstance(other, int):
            other = MapElementConstant(other)
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
        if other == 0:
            return self
        return self._op(other, operator.add)

    def __radd__(self, other: MapElement) -> 'ConditionalFunction':
        return self + other

    def __mul__(self, other: MapElement) -> 'ConditionalFunction':
        if other == 1:
            return self
        if other == 0:
            return MapElementConstant.zero
        return self._op(other, operator.mul)

    def __rmul__(self, other: MapElement) -> 'ConditionalFunction':
        return self * other

    def __sub__(self, other: MapElement) -> 'ConditionalFunction':
        if other == 0:
            return self
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

    def _simplify_with_var_values(self, var_dict: VarDict) -> 'MapElement':
        regions = []
        for condition, func in self.regions:
            condition = condition.simplify()
            if condition == FalseCondition:
                continue
            func = func._simplify_with_var_values(var_dict)
            if isinstance(condition, MapElementProcessor):
                func = condition.process(func)

            for i, (prev_cond, prev_func) in enumerate(regions):
                if prev_func == func:
                    condition_union = (prev_cond | condition)[0]
                    condition_union = condition_union.simplify()
                    regions[i][0] = condition_union
                    break
            else:
                regions.append([condition, func])

        if len(regions) == 1 and regions[0][0] == TrueCondition:
            return regions[0][1]
        return ConditionalFunction([tuple(region) for region in regions])