import operator

from typing import Dict, List, Optional, Tuple, Type, cast

from mapping_field.arithmetics import _ArithmeticMapFromFunction
from mapping_field.field import ExtElement
from mapping_field.log_utils.tree_loggers import TreeLogger, green, red, yellow
from mapping_field.mapping_field import (
    CompositionFunction, MapElement, MapElementProcessor, Var, VarDict, always_validate_promises,
)
from mapping_field.promises import IsCondition
from mapping_field.serializable import DefaultSerializable

simplify_logger = TreeLogger(__name__)


@always_validate_promises
class Condition(MapElement):
    pass

TrueCondition = None
FalseCondition = None

class BinaryCondition(Condition, DefaultSerializable):
    """
    An always True / False condition.
    """

    def __new__(cls, value: bool):
        if value:
            if TrueCondition is not None:
                return TrueCondition
        else:
            if FalseCondition is not None:
                return FalseCondition

        return super(BinaryCondition, cls).__new__(cls)

    def __init__(self, value: bool):
        super().__init__(variables=[], simplified=True)
        self.value = value
        self.promises.add_promise(IsCondition)

    def to_string(self, vars_to_str: Dict[Var, str]):
        return repr(self.value)

    def evaluate(self) -> Optional[ExtElement]:
        return 1 if self is TrueCondition else 0

    def invert(self) -> Optional[Condition]:
        return FalseCondition if (self is TrueCondition) else TrueCondition

    def and_(self, condition: MapElement):
        return condition if (self is TrueCondition) else FalseCondition

    def or_(self, condition: MapElement):
        return condition if (self is FalseCondition) else TrueCondition


TrueCondition  = BinaryCondition(True)
FalseCondition = BinaryCondition(False)


# <editor-fold desc=" ----------------------- Not Condition ----------------------- ">

@always_validate_promises
class _NotCondition(Condition, _ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__('Not', lambda a: 1-a)
        self.promises.add_promise(IsCondition)
        for v in self.vars:
            # TODO: Maybe switch directly to BoolVars?
            v.promises.add_promise(IsCondition)

    def to_string(self, vars_to_str: Dict[Var, str]):
        entries = [vars_to_str.get(v, v) for v in self.vars]
        return f'~({entries[0]})'

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v,v) for v in self.vars]

        if not isinstance(entries[0], CompositionFunction):
            return super()._simplify_with_var_values2(var_dict)
        function = entries[0].function
        comp_entries = entries[0].entries
        if function == NotCondition:
            return comp_entries[0]
        # TODO: simplify formulas like ~(a and ~b) -> ~a or b, which have fewer "not"s.

        return super()._simplify_with_var_values2(var_dict)

    def simplify(self):
        raise NotImplementedError('Delete this function')

NotCondition = _NotCondition()

def parameter_not_simplifier(var_dict: VarDict) -> Optional[MapElement]:
    entries = [var_dict[v] for v in NotCondition.vars]
    return entries[0].invert()

NotCondition.register_simplifier(parameter_not_simplifier)

MapElement.inversion = NotCondition

def _as_inversion(condition: MapElement) -> Tuple[bool, MapElement]:
    """
    return (has inversion, of elem)
    """
    if isinstance(condition, CompositionFunction) and condition.function == NotCondition:
        return True, condition.entries[0]
    return False, condition

# </editor-fold>

class _ListCondition(Condition, DefaultSerializable):
    # For intersection \ union of conditions

    AND = 0
    OR = 1

    list_classes = [cast(Type['_ListCondition'], None), cast(Type['_ListCondition'], None)]
    op_types = [operator.and_, operator.or_]
    method_names = ['and_', 'or_']
    trivials = [TrueCondition, FalseCondition]
    join_delims = ['&', '|']

    def __init_subclass__(cls, op_type: int, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.type = op_type

        _ListCondition.list_classes[op_type] = cls

        cls.op_type = cls.op_types[op_type]
        cls.join_delim = cls.join_delims[op_type]
        cls.one_condition = cls.trivials[op_type]
        cls.zero_condition = cls.trivials[1-op_type]
        setattr(cls, cls.method_names[op_type], cls.op)
        setattr(cls, cls.method_names[1 - op_type], cls.rev_op)

    def __init__(self, conditions: List[MapElement], simplified: bool = False):
        super().__init__(
            list(set(sum([condition.vars for condition in conditions],[]))),
            simplified=simplified
        )
        self.promises.add_promise(IsCondition)
        for condition in conditions:
            assert condition.has_promise(IsCondition)
        self.conditions: List[MapElement] = []

        conditions = conditions.copy()
        cls = self.__class__
        index = 0
        while index < len(conditions):
            condition = conditions[index]
            index += 1
            if isinstance(condition, cls):
                conditions.extend(condition.conditions)
                continue
            self.conditions.append(condition)

    def to_string(self, vars_to_str: Dict[Var, str]):
        entries = [vars_to_str.get(v, v) for v in self.vars]
        delim = self.__class__.join_delim
        if hasattr(self, '_binary_flag'):
            delim = delim * 2
        delim = f' {delim} '
        conditions_rep = delim.join(condition.to_string(vars_to_str) for condition in self.conditions)
        return f'[{conditions_rep}]'

    def serialization_name_conversion(self):
        return {'simplified': self._simplify_with_var_values2}

    def __eq__(self, other: MapElement) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if len(self.conditions) != len(other.conditions):
            return False

        for condition in self.conditions:
            if condition not in other.conditions:
                return False

        return True

    @classmethod
    def _op_between(cls, condition1: MapElement, condition2: MapElement) -> Optional[MapElement]:
        return cls.bin_condition[cls.type](condition1, condition2, simplify=False)._simplify2()
        # return getattr(condition1, cls.method_names[cls.type])(condition2)

    @classmethod
    def _rev_op_between(cls, condition1: MapElement, condition2: MapElement) -> Optional[MapElement]:
        rev_cls = cls.list_classes[1 - cls.type]
        return rev_cls([condition1, condition2])
        # return getattr(condition1, cls.method_names[1-cls.type])(condition2)

    def op(self, condition: MapElement) -> Optional[MapElement]:
        cls = self.__class__

        if isinstance(condition, BinaryCondition):
            # quick shortcut
            return cls._op_between(condition , self)

        if isinstance(condition, self.__class__):
            return cls([*self.conditions, *condition.conditions])

        return cls([*self.conditions, condition])

    def rev_op(self, condition: MapElement) -> Optional[MapElement]:

        cls = self.__class__
        rev_cls = cls.list_classes[1 - cls.type]

        if isinstance(condition, BinaryCondition):
            # quick shortcut
            return cls._rev_op_between(condition , self)._simplify2()

        if len(self.conditions) == 0:
            return condition

        if len(self.conditions) == 1:
            return rev_cls([self.conditions[0], condition])

        if isinstance(condition, cls):
            return cls._rev_op_against_multiple_conditions(self, condition)

        return cls._rev_op_against_single_condition(self.conditions, condition)

    @classmethod
    def _rev_op_against_multiple_conditions(cls, list_cond1: '_ListCondition', list_cond2: '_ListCondition') -> Optional[MapElement]:
        assert isinstance(list_cond1, cls) and isinstance(list_cond2, cls)
        simplify_logger.log(f'rev_op( {cls.join_delims[1-cls.type]} ) the conditions: {yellow(list_cond1)} and {yellow(list_cond2)})')
        if len(list_cond1.conditions) < len(list_cond2.conditions):
            list_cond1, list_cond2 = list_cond2, list_cond1

        conditions1 = list_cond1.conditions
        conditions2 = list_cond2.conditions

        if len(conditions2) == 0:
            return list_cond1

        # Recall the distributive law on boolean algebra gives us:
        #   ( A & B ) | ( A & C ) = A & (B | C)
        #
        # Thus, consider a union of intersections (or the corresponding intersection of unions) like:
        #   ( A & B & C & D ) | ( A & B & C0) = ( A & B & ( (C & D) | C0 )
        #
        # TODO:
        # The right hand side already contains fewer parts (5 vs 7), so I might want to consider it
        # as simpler. However, this might not be desirable, for example in cases like:
        #       ( A & B ) | ( B & C ) | ( C & A )
        # which I may want to keep as union of intersections
        #
        #
        # We have two specific cases, where I do consider this as a simpler version:
        #       1. C = C0: Full containment, where we can simply return  ->  ( A & B & C )
        #       2. (C & D) | C0 has a simplified version (e.g. maybe C0 < C, so the result is ( A & B & C0 ).

        n1 = len(conditions1)
        used_positions = [False] * n1
        special_condition = None

        # check to see if there is at most one condition in conditions2 that doesn't appear in conditions1.
        for cond2 in conditions2:
            for i in range(n1):
                if not used_positions[i] and conditions1[i] == cond2:
                    used_positions[i] = True
                    break
            else:
                if special_condition is not None:
                    return None
                special_condition = cond2

        if special_condition is None:
            # Full containment
            return list_cond2

        used_conditions = [cond1 for cond1, used in zip(conditions1, used_positions) if used]
        remaining_conditions = [cond1 for cond1, used in zip(conditions1, used_positions) if not used]
        if len(remaining_conditions) == 1:
            remaining_cond = remaining_conditions[0]
        else:   # len(...) > 1 . Cannot be 0.
            remaining_cond = cls(remaining_conditions, simplified=list_cond1.is_simplified())

        # Remark: In case (remaining_cond = list_cond1), the following call should not loop back here,
        #         because this function should only be called when the two side are lists. Unless of course
        #         someone decides to wrap special_condition into a 1 element list_cond...
        prod = cls._rev_op_between(special_condition, remaining_cond)._simplify2()
        if prod is not None:
            return cls(used_conditions + [prod]) if len(used_conditions) > 0 else prod

        return None

    @classmethod
    def _rev_op_against_single_condition(cls, conditions: List[MapElement], sp_condition: MapElement) -> Optional[MapElement]:

        # Assumption:
        # Only called when sp_condition is not an instance of this class, and this class has at least 2 conditions
        simplify_logger.log(f'rev_op( {cls.join_delims[1-cls.type]} ) the conditions vs 1: {yellow(conditions)} and {yellow(sp_condition)})')

        rev_cls = cls.list_classes[1 - cls.type]

        # Use the fact that
        #   (A | B | C) & D = (A & D) | (B & D) | (C & D)
        # If intersection with D simplified one of A,B,C, we should change them.
        # There are two possibilities:
        #       1. A & D = A_0 & D where A_0 is simpler than A, however we cannot remove the intersection with D.
        #       2. A & D = A_0     where A_0 is not computed as intersection of something with D
        #
        # If all (but one) can be written as in type (2), we return
        #       A_0 | B_0 | C_0      or     (A_0 & D) | B_0 | C_0
        # Otherwise return:
        #       ( A_0 | B_0 | C_0 ) & D

        prod_conditions   = []  # A  & D
        prod_0_conditions = []  # A0
        need_prod         = []  # True if A & D = A0 & D , False if A & D = A0
        is_simpler = False

        for cond in conditions:
            prod = cls._rev_op_between(cond, sp_condition)      # (A & D)
            simplified_prod = prod._simplify2()                 # a simplification of (A & D), or none if there is one.

            # In case where D <= A, ( iff D = A & D ) we have that (A | B | C) & D = D.
            if simplified_prod is sp_condition or simplified_prod == sp_condition:
                return sp_condition

            prod_conditions.append(simplified_prod or prod)
            if simplified_prod is None:
                prod_0_conditions.append(cond)
                need_prod.append(True)
                continue

            is_simpler = True

            # # In case where A <= D, ( iff A = A & D ) ????????
            # if (simplified_prod is cond) or (simplified_prod == cond):
            #     need_prod.append(False)
            #     continue

            if isinstance(simplified_prod, rev_cls):
                fewer_conditions = [cc for cc in simplified_prod.conditions if cc is not sp_condition]
                if len(fewer_conditions) < len(simplified_prod.conditions):
                    prod_0_conditions.append(rev_cls(fewer_conditions) if len(fewer_conditions) > 1 else fewer_conditions[0])
                    need_prod.append(True)
                    continue

            prod_0_conditions.append(simplified_prod)
            need_prod.append(False)

        if not is_simpler:
            return None

        if sum(need_prod) <= 1:
            return cls(prod_conditions)
        else:
            new_cls_condition = cls(prod_0_conditions)
            return rev_cls([new_cls_condition, sp_condition])

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
        if len(var_dict) == 0:
            if hasattr(self, '_binary_flag'):
                simplify_logger.log('Has binary flag - avoid simplifying here.')
                return None
            if self._simplified_version is self:
                return None

        cls = self.__class__

        final_conditions = []
        conditions = self.conditions.copy()

        is_whole_simpler = False

        for condition in conditions:

            simplified_condition = condition._simplify2(var_dict)
            if simplified_condition is not None:
                is_whole_simpler = True
            condition = simplified_condition or condition

            if condition is cls.zero_condition:
                return cls.zero_condition

            if condition is cls.one_condition:
                is_whole_simpler = True
                continue

            if isinstance(condition, cls):
                # unpack list condition of the same type
                is_whole_simpler = True
                conditions.extend(condition.conditions)
                continue

            # Check if this new condition intersects in a special way with an existing condition.
            # Each time this loop repeats itself, the conditions array's size must decrease by 1, so it cannot
            # continue forever.
            for existing_condition in final_conditions:
                simplify_logger.log(
                    f'Trying to combine {red(condition)} with existing {red(existing_condition)}',
                )
                # prod_cond = AndCondition(existing_condition, condition, simplify=False)._simplify2()
                # prod_cond = cls._op_between(existing_condition, condition)
                binary_op = cls([existing_condition, condition])
                binary_op._binary_flag = True
                prod_cond = binary_op._simplify2()

                if prod_cond is not None:
                    simplify_logger.log(
                        f'Combined: {red(condition)} {cls.join_delim} {red(existing_condition)}  =>  {green(prod_cond)}',
                    )
                    is_whole_simpler = True
                    final_conditions = [cond for cond in final_conditions if (cond is not existing_condition)]
                    conditions.append(prod_cond)
                    break
            else:
                # new condition cannot be intersected in a special way
                final_conditions.append(condition)

        if len(final_conditions) == 0:
            return self.__class__.one_condition

        if len(final_conditions) == 1:
            return final_conditions[0]

        if is_whole_simpler:
            return cls(final_conditions, simplified = True)

        return None

def _binary_simplify(elem: MapElement, var_dict: VarDict) -> Optional['MapElement']:
    assert isinstance(elem, _ListCondition)
    if len(elem.conditions) != 2:
        return None

    cls = elem.__class__
    rev_cls = cls.list_classes[1 - cls.type]

    cond1, cond2 = elem.conditions

    if (cond1 is cond2) or (cond1 == cond2):
        return cond1

    invert1, cond1_ = _as_inversion(cond1)
    invert2, cond2_ = _as_inversion(cond2)

    if invert1 == invert2 == True:
        return ~rev_cls([cond1_, cond2_])

    if invert1 != invert2 and ((cond1_ is cond2_) or (cond1_ == cond2_)):
        return cls.zero_condition

    method_name = cls.method_names[cls.type]
    simplify_logger.log(f'Simplify \'{method_name}\' via 1st parameter')
    result = getattr(cond1, method_name)(cond2)
    if result is not None:
        return result

    simplify_logger.log(f'Simplify \'{method_name}\' via 2nd parameter')
    return getattr(cond2, method_name)(cond1)


class IntersectionCondition(_ListCondition, MapElementProcessor, op_type = _ListCondition.AND):

    def process_function(self, func: MapElement, simplify: bool = True) -> MapElement:
        for condition in self.conditions:
            if isinstance(condition, MapElementProcessor):
                func = condition.process_function(func, simplify=simplify)
        return func

    @staticmethod
    def _binary_and_simplify(intersection_cond: MapElement, var_dict: VarDict) -> Optional['MapElement']:
        assert isinstance(intersection_cond, IntersectionCondition)
        if len(intersection_cond.conditions) != 2:
            return None

        cond1, cond2 = intersection_cond.conditions
        # TODO: use simplifier like methods
        cond1_ = cond1(condition=cond2)
        cond2_ = cond2(condition=cond1)
        if cond1_ == cond1 and cond2_ == cond2:
            return None

        if cond1_ is TrueCondition:
            return cond2_
        if cond2_ is TrueCondition:
            return cond1_
        if cond1_ is FalseCondition or cond2_ is FalseCondition:
            return FalseCondition
        return cond1_ & cond2_

IntersectionCondition.register_class_simplifier(_binary_simplify)
IntersectionCondition.register_class_simplifier(IntersectionCondition._binary_and_simplify)

MapElement.intersection = lambda cond1, cond2: IntersectionCondition([cond1, cond2]).simplify2()


class UnionCondition(_ListCondition, op_type = _ListCondition.OR):
    pass

UnionCondition.register_class_simplifier(_binary_simplify)

MapElement.union = lambda cond1, cond2: UnionCondition([cond1, cond2]).simplify2()