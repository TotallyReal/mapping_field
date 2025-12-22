import operator

from typing import Optional, cast

from mapping_field.associative import AssociativeListFunction
from mapping_field.field import ExtElement
from mapping_field.log_utils.tree_loggers import TreeLogger, green, log_context, magenta, yellow
from mapping_field.mapping_field import (
    CompositeElementFromFunction, MapElement, MapElementProcessor, OutputProperties,
    SimplifierOutput, Var, class_simplifier, simplifier_context,
)
from mapping_field.property_engines import is_condition
from mapping_field.utils.processors import ProcessFailureReason
from mapping_field.utils.serializable import DefaultSerializable

simplify_logger = TreeLogger(__name__)


class BinaryCondition(MapElement, DefaultSerializable):
    """
    An always True / False condition.
    """
    _conditions: dict[bool, Optional['BinaryCondition']] = {True: None, False: None}

    def __new__(cls, value: bool):
        condition = BinaryCondition._conditions[value]
        if condition is None:
            condition = super(BinaryCondition, cls).__new__(cls)
            BinaryCondition._conditions[value] = condition

        return condition

    def __init__(self, value: bool):
        super().__init__(variables=[], simplified=True)
        self.value = value

    def to_string(self, vars_to_str: dict[Var, str]):
        return str(self.value)

    def evaluate(self) -> ExtElement | None:
        return 1 if self is TrueCondition else 0

    def invert(self) -> 'BinaryCondition':
        return FalseCondition if (self is TrueCondition) else TrueCondition

    def and_(self, condition: MapElement):
        return condition if (self is TrueCondition) else FalseCondition

    def or_(self, condition: MapElement):
        return condition if (self is FalseCondition) else TrueCondition
is_condition.add_auto_class(BinaryCondition)

TrueCondition  = BinaryCondition(True)
FalseCondition = BinaryCondition(False)


# <editor-fold desc=" ----------------------- Not Condition ----------------------- ">


class _NotCondition(CompositeElementFromFunction):

    def __init__(self, operand: MapElement | None = None):
        operands = [operand] if operand is not None else None
        super().__init__(operands=operands, name="Not", function=lambda a: 1 - a)

        # TODO: make sure that the variable is a condition variable
        # for v in self.vars:
        #     # TODO: Maybe switch directly to BoolVars?
        #     v.promises.add_promise(IsCondition)

    @property
    def operand(self) -> MapElement:
        return self.operands[0]

    @operand.setter
    def operand(self, value: MapElement):
        self.operands[0] = value

    def to_string(self, vars_to_str: dict[Var, str]):
        return f"~({self.operand.to_string(vars_to_str)})"

    def __eq__(self, other):
        return isinstance(other, _NotCondition) and self.operand == other.operand

    __hash__ = MapElement.__hash__

    def _simplify_with_var_values(self) -> MapElement | None:
        operand = self.operand

        if isinstance(operand, BinaryCondition):
            return operand.invert()
        if isinstance(operand, _NotCondition):
            return operand.operand

        # TODO: simplify formulas like ~(a and ~b) -> ~a or b, which have fewer "not"s ?

        return super()._simplify_with_var_values()

    @class_simplifier
    @staticmethod
    def _to_inversion_simplifier(inversion_func: MapElement) -> MapElement | None:
        """
            run elem.invert()
        """
        assert isinstance(inversion_func, _NotCondition)
        return inversion_func.operand.invert()

is_condition.add_auto_class(_NotCondition)
NotCondition = _NotCondition()
MapElement.inversion = NotCondition


def _as_inversion(condition: MapElement) -> tuple[bool, MapElement]:
    """
    return (has inversion, of elem)
    """
    if isinstance(condition, _NotCondition):
        return True, condition.operand
    return False, condition


# </editor-fold>


class _ListCondition(AssociativeListFunction, DefaultSerializable):
    # For intersection \ union of conditions

    AND = 0
    OR = 1
    left_bracket  = "["
    right_bracket = "]"

    list_classes    = [cast(type["_ListCondition"], None), cast(type["_ListCondition"], None)]
    op_types        = [operator.and_, operator.or_]
    method_names    = ["and_", "or_"]
    trivials        = [TrueCondition, FalseCondition]
    op_symbols      = ["&", "|"]

    def __init_subclass__(cls, op_type: int, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.type = op_type

        _ListCondition.list_classes[op_type] = cls

        cls.op_type = cls.op_types[op_type]
        cls.op_symbol = cls.op_symbols[op_type]
        cls.one_condition = cls.trivials[op_type]
        cls.trivial_element = cls.trivials[op_type]
        cls.zero_condition = cls.trivials[1 - op_type]
        cls.final_element = cls.trivials[1 - op_type]
        setattr(cls, cls.method_names[op_type], cls.op)
        setattr(cls, cls.method_names[1 - op_type], cls.rev_op)

    @classmethod
    def rev_class(cls):
        return cls.list_classes[1 - cls.type]

    @classmethod
    def _unpack_list(cls, elements: list[MapElement]) -> list[MapElement]:
        elements = elements.copy()
        all_elements = []

        while len(elements)>0:
            element = elements.pop(0)
            if isinstance(element, cls):
                elements.extend(element.operands)
                continue
            all_elements.append(element)
        return all_elements

    def __init__(self, operands: list[MapElement], simplified: bool = False,
            output_properties: OutputProperties | None = None):
        for operand in operands:
            assert is_condition.compute(operand, simplifier_context), f"{operand} is not a condition"

        super().__init__(
            operands=self.__class__._unpack_list(operands),
            simplified=simplified,
            output_properties=output_properties
        )


    @property
    def conditions(self) -> list[MapElement]:
        return self.operands

    @conditions.setter
    def conditions(self, value: list[MapElement]):
        self.operands = value

    def serialization_name_conversion(self):
        return {"simplified": self._simplify_with_var_values}

    @classmethod
    def _op_between(cls, condition1: MapElement, condition2: MapElement) -> MapElement | None:
        return cls.bin_condition[cls.type](condition1, condition2, simplify=False)._simplify()

    @classmethod
    def _rev_op_between(cls, condition1: MapElement, condition2: MapElement) -> MapElement | None:
        return cls.rev_class()([condition1, condition2])

    def op(self, condition: MapElement) -> MapElement | None:
        cls = self.__class__

        if isinstance(condition, BinaryCondition):
            # quick shortcut
            return cls._op_between(condition, self)

        if isinstance(condition, self.__class__):
            return cls([*self.conditions, *condition.conditions])

        return cls([*self.conditions, condition])

    def rev_op(self, condition: MapElement) -> MapElement | None:

        cls = self.__class__
        rev_cls = cls.rev_class()

        if isinstance(condition, BinaryCondition):
            # quick shortcut
            return cls._rev_op_between(condition, self)._simplify()

        if len(self.conditions) == 0:
            return condition

        if len(self.conditions) == 1:
            return rev_cls([self.conditions[0], condition])

        if isinstance(condition, cls):
            return cls._rev_op_against_multiple_conditions(self, condition)

        return cls._rev_op_against_single_condition(self.conditions, condition)

    @classmethod
    def _rev_op_against_multiple_conditions(
        cls, list_cond1: "_ListCondition", list_cond2: "_ListCondition"
    ) -> MapElement | None:
        assert isinstance(list_cond1, cls) and isinstance(list_cond2, cls)
        simplify_logger.log(
            f"rev_op( {cls.op_symbols[1 - cls.type]} ) the conditions: {yellow(list_cond1)} and {yellow(list_cond2)})"
        )
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
        else:  # len(...) > 1 . Cannot be 0.
            remaining_cond = cls(remaining_conditions, simplified=list_cond1.is_simplified())

        # Remark: In case (remaining_cond = list_cond1), the following call should not loop back here,
        #         because this function should only be called when the two side are lists. Unless of course
        #         someone decides to wrap special_condition into a 1 element list_cond...
        prod = cls._rev_op_between(special_condition, remaining_cond)._simplify()
        if prod is not None:
            return cls(used_conditions + [prod]) if len(used_conditions) > 0 else prod

        return None

    @classmethod
    def _rev_op_against_single_condition(
        cls, conditions: list[MapElement], sp_condition: MapElement
    ) -> MapElement | None:

        # Assumption:
        # Only called when sp_condition is not an instance of this class, and this class has at least 2 conditions
        simplify_logger.log(
            f"rev_op( {cls.op_symbols[1 - cls.type]} ) the conditions vs 1: {yellow(conditions)} and {yellow(sp_condition)})"
        )

        rev_cls = cls.rev_class()

        # Use the fact that
        #   (A | B | C) & D = (A & D) | (B & D) | (C & D)
        # If intersection with D simplifies one of A,B,C, we might want to use it.
        # There are two possibilities:
        #       1. A & D = A_0 & D where A_0 is simpler than A, however we cannot remove the intersection with D.
        #       2. A & D = A_0     where A_0 is not computed as intersection of something with D.
        # In any case we get that :
        #   (A | B | C) & D = (A0 & D) | (B0 & D) | (C0 & D) = (A0 | B0 | C0) & D
        # We want to choose between the 2nd and 3rd expression as possible simplifications.
        #
        # The simplest case is if A & D = D , since then
        #       (A & D) | (B & D) | (C & D) = D | (B & D) | (C & D) = D
        #
        # The second case, is if when there is a simplification, it is because D is larger. For example if
        # both B & D and C & D cannot be simplified, and A & D = A, then the expression from above are:
        #       (A | B | C) & D = A | (B & D) | (C & D) = (A | B | C) & D
        # The middle expression is not 'simpler' than the original, and the right expression is exactly the original.
        # In this case, there is no simplification. However, if B & D = B and C & D = C as well, then we can simplify
        # it to be just
        #       ( A | B | C )
        #
        # Finally, assume that A & D = A0 & D with A0 different from both A and D, then:
        #
        # 1. If all (but one) can be written as in type (2), we return
        #       A_0 | B_0 | C_0      or     (A_0 & D) | B_0 | C_0  ,
        # 2. Otherwise, return:
        #       ( A_0 | B_0 | C_0 ) & D

        prod_conditions   = []  # A & D = A0 & D
        prod_0_conditions = []  # A0
        need_prod         = []  # True if A & D != A0, namely we must write it as A & D = A0 & D
        contain_count     = 0   # Count number of times where A & D = A
        is_simpler = False

        for cond in conditions:
            prod = cls._rev_op_between(cond, sp_condition)      # (A & D)
            simplified_prod = prod._simplify()                 # a simplification of (A & D), or none if there is one.

            # In case where D <= A, ( iff D = A & D ) we have that (A | B | C) & D = D.
            if simplified_prod is sp_condition or simplified_prod == sp_condition:
                return sp_condition

            prod_conditions.append(simplified_prod or prod)
            if simplified_prod is None:
                # Can't write A & D in a simpler way
                prod_0_conditions.append(cond)
                need_prod.append(True)
                continue

            # In case where A <= D, so that A & D = A
            if (simplified_prod is cond) or (simplified_prod == cond):
                contain_count += 1
                need_prod.append(False)
                continue

            is_simpler = True

            if isinstance(simplified_prod, rev_cls):
                fewer_conditions = [cc for cc in simplified_prod.conditions if cc is not sp_condition]
                if len(fewer_conditions) < len(simplified_prod.conditions):
                    prod_0_conditions.append(
                        rev_cls(fewer_conditions) if len(fewer_conditions) > 1 else fewer_conditions[0]
                    )
                    need_prod.append(True)
                    continue

            prod_0_conditions.append(simplified_prod)
            need_prod.append(False)

        if contain_count == len(conditions):
            return cls(conditions)

        if not is_simpler:
            return None

        # # There is one other case we should ignore when "simplifying".
        # # If A, B_s, B_l are conditions with B_s < B_l, then
        # #       (A & B_l) | B_s = (A | B_s) & (B_l | B_s) = (A | B_s) & B_l
        # # If we simplify in one direction, we will have to simplify in the other direction as well, which will
        # # cause a loop.
        #
        # if sum(need_prod) == 1 and len(conditions) == 2:
        #     index = 0 if need_prod[1] else 1
        #     # TODO: The following line is not that good, since the == operator doesn't always return true
        #     #       even when the conditions are the same.
        #     #       If it causes too many problems in the future, we can ignore this case whole together, and just
        #     #       return None.
        #     if prod_conditions[index] == conditions[index]:    # flag is false exactly once
        #         return None


        if sum(need_prod) <= 1:
            # Almost all products are simplified
            return cls(prod_conditions)
        else:
            new_cls_condition = cls(prod_0_conditions)
            return rev_cls([new_cls_condition, sp_condition])

    def invert(self) -> MapElement | None:
        """
                ~ (x & ~y) => ~x | y
            Only works if this doesn't increase the total number of inversion signs.
        """
        # TODO: add tests
        invert_counter = sum([isinstance(operand, _NotCondition) for operand in self.operands], 0)
        if len(self.operands) - 1 <= 2 * invert_counter:
            rev_cls = self.__class__.rev_class()
            return rev_cls([~operand for operand in self.operands])
        return None

    @class_simplifier
    @staticmethod
    def _invert_extractor_simplifier(list_condition: MapElement) -> SimplifierOutput:
        """
                (x & ~y & ~z & ~w) => ~(~x | y | z | w)
            Only works if this strictly reduces the total number of minus signs.
        """
        assert isinstance(list_condition, _ListCondition)
        if _ListCondition.is_binary(list_condition):
            return ProcessFailureReason('Not applicable to the binary version', trivial=True)
        invert_counter = sum([isinstance(operand, _NotCondition) for operand in list_condition.operands], 0)
        if len(list_condition.operands)+1 < 2*invert_counter:
            rev_cls = list_condition.__class__.rev_class()
            return NotCondition(rev_cls([~operand for operand in list_condition.operands]))
        return ProcessFailureReason('Not enough inversions to extract', trivial = True)

    @class_simplifier
    @staticmethod
    def _unpack_inversions_simplifier(list_condition: MapElement) -> SimplifierOutput:
        """
                x & ~(y | z)   =>      x & (~y) & (~z)
            Unlike just inversion of _ListCondition, where we simplify only when the number of inversions decrease,
            as a partial list we always open the brackets.
        """
        assert isinstance(list_condition, _ListCondition)
        non_inversion_operands = []
        inversion_operands = []
        cls = list_condition.__class__
        rev_cls = cls.rev_class()
        for operand in list_condition.operands:
            if isinstance(operand, _NotCondition) and isinstance(operand.operand, rev_cls):
                for sub_operand in operand.operand.operands:
                    inversion_operands.append(~sub_operand)
            else:
                non_inversion_operands.append(operand)
        if len(inversion_operands) > 0:
            return cls(non_inversion_operands + inversion_operands)
        return ProcessFailureReason('No inversion signs', trivial = True)

    @class_simplifier
    @staticmethod
    def _binary_simplify(elem: MapElement) -> MapElement | None:
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

        # if invert1 == invert2 == True:
        #     return ~rev_cls([cond1_, cond2_])

        if invert1 != invert2 and ((cond1_ is cond2_) or (cond1_ == cond2_)):
            return cls.zero_condition

        method_name = cls.method_names[cls.type]

        with log_context(simplify_logger, start_msg=f"Simplify '{method_name}' via 1st parameter") as log_result:
            result = getattr(cond1, method_name)(cond2)
            if result is not None:
                log_result.set(green(result))
                return result
            log_result.set(magenta(' # # # '))

        with log_context(simplify_logger, start_msg=f"Simplify '{method_name}' via 2st parameter") as log_result:
            result = getattr(cond2, method_name)(cond1)
            if result is not None:
                log_result.set(green(result))
                return result
            log_result.set(magenta(' # # # '))

        return None

is_condition.add_auto_class(_ListCondition)


class IntersectionCondition(_ListCondition, MapElementProcessor, op_type=_ListCondition.AND):

    def process_function(self, func: MapElement, simplify: bool = True) -> MapElement:
        for condition in self.conditions:
            if isinstance(condition, MapElementProcessor):
                func = condition.process_function(func, simplify=simplify)
        return func

    @class_simplifier
    @staticmethod
    def _binary_and_simplify(intersection_cond: MapElement) -> MapElement | None:
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


MapElement.intersection = lambda cond1, cond2: IntersectionCondition([cond1, cond2]).simplify()


class UnionCondition(_ListCondition, op_type=_ListCondition.OR):
    pass


MapElement.union = lambda cond1, cond2: UnionCondition([cond1, cond2]).simplify()
