from abc import abstractmethod
from typing import List, Tuple, Optional
import operator
import functools

from mapping_field import MapElement, Var, VarDict, FuncDict, ExtElement, MapElementConstant

def _param_to_condition(f):
    @functools.wraps(f)
    def wrapper(self, other):
        if isinstance(other, bool):
            other = TrueCondition if other else FalseCondition

        if not isinstance(other, Condition):
            return NotImplemented

        return f(self, other)

    return wrapper


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

    @_param_to_condition
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
        condition, is_simpler = condition1.and_simpler(condition2)
        if is_simpler:
            return condition, is_simpler
        return condition2 & condition1

    def and_simpler(self, condition: 'Condition') -> Tuple['Condition', bool]:
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
            return condition.and_simpler(self)

        if self == condition:
            return self, True

        return ConditionIntersection([self, condition]), False

    @_param_to_condition
    def __and__(self, condition) -> 'Condition':
        return self.and_simpler(condition)[0]

    @classmethod
    def union(cls, condition1: 'Condition', condition2: 'Condition') -> Tuple['Condition', bool]:
        """
        Similar to Condition.intersection(...), but with union.
        """
        condition, is_simpler = condition1.or_simpler(condition2)
        if is_simpler:
            return condition, is_simpler
        return condition2.or_simpler(condition1)

    def or_simpler(self, condition: 'Condition') -> Tuple['Condition', bool]:
        """
        Similar to and_simpler, but for __or__
        """
        if isinstance(condition, BinaryCondition) or isinstance(condition, ConditionUnion):
            return condition.or_simpler(self)

        if self == condition:
            return self, True

        return ConditionUnion([self, condition]), False

    @_param_to_condition
    def __or__(self, condition) -> 'Condition':
        """
        Similar to __and__, but for __or__
        """
        return self.or_simpler(condition)[0]

    def __mul__(self, condition: 'Condition'):
        """
        Same as __and__
        """
        return self & condition

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

    def and_simpler(self, condition: Condition) -> Tuple[Condition, bool]:
        return (condition if self.value else FalseCondition), True

    def or_simpler(self, condition: 'Condition') -> Tuple['Condition', bool]:
        return (condition if not self.value else TrueCondition), True

TrueCondition  = BinaryCondition(True)
FalseCondition = BinaryCondition(False)


class MapElementProcessor:

    @abstractmethod
    def process_function(self, func: MapElement) -> MapElement:
        pass

class _ListCondition(Condition):
    # For intersection \ union of conditions

    AND = 0
    OR = 1

    def __init_subclass__(cls, op_type: int):
        cls.type = op_type

        cls.op_types = [operator.and_, operator.or_]
        cls.method_names = ['and_simpler', 'or_simpler']
        cls.trivials = [TrueCondition, FalseCondition]
        cls.join_delims = [' & ', ' | ']

        cls.op_type = cls.op_types[op_type]
        cls.join_delim = cls.join_delims[op_type]
        cls.one_condition = cls.trivials[op_type]
        cls.zero_condition = cls.trivials[1-op_type]
        setattr(cls, cls.method_names[op_type], cls.op)
        setattr(cls, cls.method_names[1 - op_type], cls.rev_op)

    def __init__(self, conditions: List[Condition], simplified: bool = False):
        super().__init__(
            list(set(sum([condition.vars for condition in conditions],[])))
        )
        self.conditions = conditions
        self._simplified = simplified

    def __repr__(self):
        conditions_rep = self.__class__.join_delim.join(repr(condition) for condition in self.conditions)
        return f'[{conditions_rep}]'

    @classmethod
    def _op_simpler_between(cls, condition1: Condition, condition2: Condition) -> Tuple[Condition, bool]:
        return getattr(condition1, cls.method_names[cls.type])(condition2)

    @classmethod
    def _rev_op_simpler_between(cls, condition1: Condition, condition2: Condition) -> Tuple[Condition, bool]:
        return getattr(condition1, cls.method_names[1-cls.type])(condition2)

    def op(self, condition: Condition) -> Tuple[Condition, bool]:
        cls = self.__class__

        if isinstance(condition, BinaryCondition):
            return cls._op_simpler_between(condition , self)

        if isinstance(condition, self.__class__):
            return cls([*self.conditions, *condition.conditions]), False

        return cls([*self.conditions, condition]), False

    def rev_op(self, condition) -> Tuple[Condition, bool]:

        cls = self.__class__

        if isinstance(condition, BinaryCondition):
            return cls._rev_op_simpler_between(condition , self)

        conditions1 = self.conditions.copy()
        conditions2 = condition.conditions.copy() if isinstance(condition, cls) else [condition]

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
                    return cls._rev_op_simpler_between(super(), condition)
                special_condition = cond2

        if special_condition is None:
            # Full containement
            return (self if swapped else condition), True

        remaining_conditions = [cond1 for cond1, used in zip(conditions1, used_positions) if not used]
        if len(remaining_conditions) == 1:
            second_special_condition = remaining_conditions[0]

            prod, is_simpler = cls._rev_op_simpler_between(special_condition, second_special_condition)
            if is_simpler:
                conditions = [c for flag, c in zip(used_positions, conditions1) if flag]
                return cls(conditions + [prod]), True
        elif len(remaining_conditions) < n1: # there was some simplification
            prod, is_simpler = cls._rev_op_simpler_between(special_condition, cls(remaining_conditions))
            if is_simpler:
                conditions = [c for flag, c in zip(used_positions, conditions1) if flag]
                return cls(conditions + [prod]), True


        return cls._rev_op_simpler_between(super(), condition)

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
                    prod_cond, is_simpler = cls._op_simpler_between(existing_condition, condition)
                    if not is_simpler:
                        prod_cond, is_simpler = cls._op_simpler_between(condition, existing_condition)

                    if is_simpler:
                        final_conditions = [cond for cond in final_conditions if (cond is not existing_condition)]
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

class ConditionIntersection(_ListCondition, MapElementProcessor, op_type = _ListCondition.AND):

    def __init__(self, conditions: List[Condition], simplified: bool = False):
        super().__init__(conditions, simplified)

    def process_function(self, func: MapElement) -> MapElement:
        for condition in self.conditions:
            if isinstance(condition, MapElementProcessor):
                func = condition.process_function(func)
        return func


class ConditionUnion(_ListCondition, op_type = _ListCondition.OR):

    def __init__(self, conditions: List[Condition], simplified: bool = False):
        super().__init__(conditions, simplified)
