import operator
from typing import Optional, List, Tuple

from mapping_field.arithmetics import _ArithmeticMapFromFunction
from mapping_field.field import ExtElement
from mapping_field.mapping_field import MapElement, VarDict, CompositionFunction, OutputPromise, \
    always_validate_promises
from mapping_field.serializable import DefaultSerializable
from mapping_field.log_utils.tree_loggers import TreeLogger, red, green

simplify_logger = TreeLogger(__name__)

IsCondition = OutputPromise("Condition")

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
        super().__init__(variables=[])
        self.value = value
        self.add_promise(IsCondition)

    def to_string(self, entries: List[str]):
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
        self.add_promise(IsCondition)
        for v in self.vars:
            # TODO: Maybe switch directly to BoolVars?
            v.add_promise(IsCondition)

    def to_string(self, entries: List[str]):
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

# <editor-fold desc=" ----------------------- And Condition ----------------------- ">

@always_validate_promises
class _AndCondition(Condition, _ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__('And', lambda a, b: a * b)
        self.add_promise(IsCondition)
        for v in self.vars:
            # TODO: Maybe switch directly to BoolVars?
            v.add_promise(IsCondition)

    def to_string(self, entries: List[str]):
        return f'({entries[0]} & {entries[1]})'


AndCondition = _AndCondition()

def parameter_and_simplifier(var_dict: VarDict) -> Optional[MapElement]:
    entries = [var_dict[v] for v in AndCondition.vars]

    invert0, cond0 = _as_inversion(entries[0])
    invert1, cond1 = _as_inversion(entries[1])

    if invert0 == invert1 == True:
        return ~(cond0 | cond1)

    if invert0 != invert1 and ((cond0 is cond1) or (cond0 == cond1)):
        return FalseCondition

    if (entries[0] is entries[1]) or (entries[0] == entries[1]):
        return entries[0]

    simplify_logger.log('Simplify \'and\' via 1st parameter')
    result = entries[0].and_(entries[1])
    if result is not None:
        return result

    simplify_logger.log('Simplify \'and\' via 2nd parameter')
    return entries[1].and_(entries[0])
AndCondition.register_simplifier(parameter_and_simplifier)

def associative_and_simplifier(var_dict: VarDict) -> Optional[MapElement]:
    entry0, entry1 = [var_dict[v] for v in AndCondition.vars]
    if isinstance(entry0, CompositionFunction) and (entry0.function is AndCondition):
        return IntersectionCondition([*(entry0.entries), entry1])
    if isinstance(entry1, CompositionFunction) and (entry1.function is AndCondition):
        return IntersectionCondition([entry0, *(entry1.entries)])
    return None
AndCondition.register_simplifier(associative_and_simplifier)

MapElement.intersection = AndCondition

# </editor-fold>

# <editor-fold desc=" ----------------------- Or Condition ----------------------- ">

@always_validate_promises
class _OrCondition(Condition, _ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__('Or', lambda a, b: a + b - a * b)
        self.add_promise(IsCondition)
        for v in self.vars:
            # TODO: Maybe switch directly to BoolVars?
            v.add_promise(IsCondition)

    def to_string(self, entries: List[str]):
        return f'({entries[0]} | {entries[1]})'

OrCondition = _OrCondition()

def parameter_or_simplifier(var_dict: VarDict) -> Optional[MapElement]:
    entries = [var_dict[v] for v in OrCondition.vars]

    invert0, cond0 = _as_inversion(entries[0])
    invert1, cond1 = _as_inversion(entries[1])

    if invert0 == invert1 == True:
        return ~(cond0 & cond1)

    if invert0 != invert1 and ((cond0 is cond1) or (cond0 == cond1)):
        return TrueCondition

    if (entries[0] is entries[1]) or (entries[0] == entries[1]):
        return entries[0]
    simplify_logger.log('Simplify \'or\' via 1st parameter')
    result = entries[0].or_(entries[1])
    if result is not None:
        return result
    simplify_logger.log('Simplify \'or\' via 2nd parameter')
    return entries[1].or_(entries[0])
OrCondition.register_simplifier(parameter_or_simplifier)

def associative_or_simplifier(var_dict: VarDict) -> Optional[MapElement]:
    entry0, entry1 = [var_dict[v] for v in OrCondition.vars]
    if isinstance(entry0, CompositionFunction) and (entry0.function is OrCondition):
        return UnionCondition([*(entry0.entries), entry1])
    if isinstance(entry1, CompositionFunction) and (entry1.function is OrCondition):
        return UnionCondition([entry0, *(entry1.entries)])
    return None
OrCondition.register_simplifier(associative_or_simplifier)

MapElement.union = OrCondition

# </editor-fold>

class _ListCondition(Condition):
    # For intersection \ union of conditions

    AND = 0
    OR = 1

    bin_condition = [AndCondition, OrCondition]
    list_classes = [cast(Type['_ListCondition'], None), cast(Type['_ListCondition'], None)]
    op_types = [operator.and_, operator.or_]
    method_names = ['and_', 'or_']
    trivials = [TrueCondition, FalseCondition]
    join_delims = [' & ', ' | ']

    def __init_subclass__(cls, op_type: int, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.type = op_type

        _ListCondition.list_classes[op_type] = cls

        cls.op_type = cls.op_types[op_type]
        cls.join_delim = cls.join_delims[op_type]
        cls.one_condition = cls.trivials[op_type]
        cls.zero_condition = cls.trivials[1-op_type]
        setattr(cls, cls.method_names[op_type], cls.op)
        # setattr(cls, cls.method_names[1 - op_type], cls.rev_op)

    def __init__(self, conditions: List[MapElement], simplified: bool = False):
        super().__init__(
            list(set(sum([condition.vars for condition in conditions],[])))
        )
        self.add_promise(IsCondition)
        self._simplified = simplified
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

    def to_string(self, vars_str_list: List[str]):
        conditions_rep = self.__class__.join_delim.join(condition.to_string(vars_str_list) for condition in self.conditions)
        return f'[{conditions_rep}]'

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

    def op(self, condition: MapElement) -> Optional[MapElement]:
        cls = self.__class__

        if isinstance(condition, BinaryCondition):
            # quick shortcut
            return cls._op_between(condition , self)

        if isinstance(condition, self.__class__):
            return cls([*self.conditions, *condition.conditions])

        return cls([*self.conditions, condition])

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
        if self._simplified:
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

            if condition is cls.one_condition:
                is_whole_simpler = True
                continue

            if isinstance(condition, cls):
                # unpack list condition of the same type
                is_whole_simpler = True
                conditions.extend(condition.conditions)
                continue

            while (True):
                # Check if this new condition intersects in a special way with an existing condition.
                # Each time this loop repeats itself, the conditions array's size must decrease by 1, so it cannot
                # continue forever.
                if condition is cls.zero_condition:
                    return cls.zero_condition

                for existing_condition in final_conditions:
                    simplify_logger.log(
                        f'Trying to combine {red(condition)} with existing {red(existing_condition)}',
                    )
                    # prod_cond = AndCondition(existing_condition, condition, simplify=False)._simplify2()
                    prod_cond = cls._op_between(existing_condition, condition)
                    # if prod_cond is None:
                    #     prod_cond = cls._op_between(condition, existing_condition)

                    if prod_cond is not None:
                        simplify_logger.log(
                            f'Combined: {red(condition)} {cls.join_delim} {red(existing_condition)}  =>  {green(prod_cond)}',
                        )
                        is_whole_simpler = True
                        final_conditions = [cond for cond in final_conditions if (cond is not existing_condition)]
                        break

                else:
                    # new condition cannot be intersected in a special way
                    final_conditions.append(condition)
                    break

        if len(final_conditions) == 0:
            return self.__class__.one_condition

        if len(final_conditions) == 1:
            return final_conditions[0]

        if is_whole_simpler:
            return cls(final_conditions, simplified = True)

        return None

class IntersectionCondition(_ListCondition, op_type = _ListCondition.AND):
    pass

class UnionCondition(_ListCondition, op_type = _ListCondition.OR):
    pass
