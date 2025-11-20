from typing import Optional

from mapping_field.arithmetics import BinaryCombination, as_neg
from mapping_field.conditions import (
    FalseCondition, IntersectionCondition, TrueCondition, UnionCondition,
)
from mapping_field.linear import Linear
from mapping_field.log_utils.tree_loggers import TreeLogger, red
from mapping_field.mapping_field import (
    CompositeElement, ExtElement, MapElement, MapElementConstant, SimplifierOutput, Var,
    convert_to_map,
)
from mapping_field.promises import IsCondition, IsIntegral
from mapping_field.ranged_condition import BoolVar, InRange, IntervalRange, RangeCondition
from mapping_field.utils.processors import ProcessFailureReason
from mapping_field.utils.serializable import DefaultSerializable

# from mapping_field.new_code.linear import LinearTransformer, Linear

simplify_logger = TreeLogger(__name__)


def _two_power(k):
    k = abs(k)
    m = 0
    while k % 2 == 0:
        k /= 2
        m += 1
    return m


#
class BinaryExpansion(CompositeElement, DefaultSerializable):

    auto_promises = [IsIntegral]

    @classmethod
    def of(cls, map_elem: MapElement) -> Optional["BinaryExpansion"]:
        if isinstance(map_elem, BinaryExpansion):
            return map_elem
        if isinstance(map_elem, BoolVar):
            return BinaryExpansion([map_elem])
        # if isinstance(map_elem, BoundedIntVar) and map_elem.max_value - map_elem.min_value == 2:
        #     return BinaryExpansion([BoolVar(f'{map_elem.name}_bool')]), map_elem.min_value

        return None

    @staticmethod
    def generate(var_name: str, num_digits: int):
        num_digits = max(num_digits, 1)
        for i in range(num_digits):
            assert Var.try_get(f"{var_name}_{i}") is None

        return BinaryExpansion([BoolVar(f"{var_name}_{i}") for i in range(num_digits)])

    @staticmethod
    def convert_coefficients(coefficients: list) -> list[MapElementConstant | BoolVar]:
        converted_coefficients = []
        for c in coefficients:
            c = convert_to_map(c)

            value = c.evaluate()
            if value is not None:
                assert c in (0,1), f"Only possible integers are 0 or 1, instead got {c}"
                converted_coefficients.append(MapElementConstant.zero if c == 0 else MapElementConstant.one)
                continue

            assert c.has_promise(IsCondition), (
                f"Coefficients for Binary expansion must be conditions (0\\1 valued). "
                f"Instead got {c} of type {c.__class__}"
            )
            converted_coefficients.append(c)
        return converted_coefficients

    def __init__(self, coefficients: list[int | MapElementConstant | BoolVar]):
        """
        Represents an integer number in a binary expansion:

            sum_i coef[i] * 2^i

        """
        # For now, each bool variable can appear at most once
        super().__init__(operands=BinaryExpansion.convert_coefficients(coefficients))
        self._compute_range()

    def _compute_range(self):

        self._constant = 0                  # The value of this map element without the bool variables
        self._bool_max_value = [0]          # max value of only bool variables, up to position i-1

        two_power = 1
        for i, cc in enumerate(self.coefficients):
            value = cc.evaluate()
            if value is not None:
                self._constant += two_power * value
                self._bool_max_value.append(self._bool_max_value[-1])
            else:
                self._bool_max_value.append(self._bool_max_value[-1] + two_power)
            two_power *= 2
        self.promises.add_promise(InRange(IntervalRange[self._constant, self._constant + self._bool_max_value[-1]]))

    @property
    def coefficients(self) -> list[MapElement]:
        return self.operands

    @coefficients.setter
    def coefficients(self, value: list[MapElement]):
        self.operands = BinaryExpansion.convert_coefficients(value)

    def copy_with_operands(self, operands: list[MapElement]) -> MapElement:
        copy_version = super().copy_with_operands(operands)
        assert isinstance(copy_version, BinaryExpansion)
        copy_version._compute_range()
        return copy_version

    def to_string(self, vars_to_str: dict[Var, str]):
        indices = [i for i, v in enumerate(self.coefficients) if v != 0]
        if len(indices) == 0:
            return "0"
        str_coefficients = [c.to_string(vars_to_str) for c in self.coefficients]
        if indices[-1] == 0:
            return str_coefficients[0]
        vars_str = ", ".join([str(v) for v in str_coefficients[: 1 + indices[-1]]])
        return f"Bin[{vars_str}]"

    def evaluate(self) -> ExtElement | None:
        return self._constant if (self._bool_max_value[-1] == 0) else None

    def split_constant(self) -> tuple[Optional["BinaryExpansion"], MapElementConstant]:
        constant_part = MapElementConstant(self._constant)
        if self._constant == 0:
            return self, constant_part

        if self._bool_max_value[-1] == 0:
            return None, constant_part

        coefs = [0 if c.evaluate() is not None else c for c in self.coefficients]
        return BinaryExpansion(coefs), constant_part

    def __eq__(self, other):

        value2 = other.evaluate() if isinstance(other, MapElement) else other
        value1 = self.evaluate()
        if value1 is not None:
            return value1 == value2

        if not isinstance(other, BinaryExpansion):
            return super().__eq__(other)

        coef1 = self.coefficients
        n1 = len(coef1)
        coef2 = other.coefficients
        n2 = len(coef2)

        n = min(n1, n2)

        return (
            all([c1 == c2 for c1, c2 in zip(coef1, coef2)])
            and all([c1 == 0 for c1 in coef1[n:]])
            and all([c2 == 0 for c2 in coef2[n:]])
        )

    #
    #     def as_assignment(self, k: int) -> Condition:
    #         return RangeCondition(self, (k, k+1)).simplify()
    #
    #     # <editor-fold desc=" ------------------------ Arithmetic ------------------------">
    #
    @staticmethod
    def linear_combination(k1: int, elem1: MapElement, k2: int, elem2: MapElement) -> tuple[int, MapElement] | None:
        if not (isinstance(elem1, BinaryExpansion) and isinstance(elem2, BinaryExpansion)):
            return None

        simplify_logger.log(
            f"Trying to linear combine to Binary Expansion {red(k1)} * {red(elem1)} + {red(k2)} * {red(elem2)}"
        )

        coef1 = elem1.coefficients
        non_zero1 = [i for i, v in enumerate(coef1) if v != 0]
        if k1 == 0 or len(non_zero1) == 0:
            return k2, elem2

        coef2 = elem2.coefficients
        non_zero2 = [i for i, v in enumerate(coef2) if v != 0]
        if k2 == 0 or len(non_zero2) == 0:
            return k1, elem1

        # Check if both elements are the same up to a power of 2
        if len(non_zero1) == len(non_zero2):
            diff = non_zero1[0] - non_zero2[0]
            if all(((coef1[i1] == coef2[i2]) and (i1 - i2 == diff)) for i1, i2 in zip(non_zero1, non_zero2)):
                if diff >= 0:
                    return (2**diff) * k1 + k2, elem2
                else:
                    return (2 ** (-diff)) * k2 + k1, elem1

        # Under assumption, k1 and k2 are coprime

        m1 = _two_power(k1)
        m2 = _two_power(k2)
        if abs(k1) != 2 ** m1 or abs(k2) != 2**m2:
            return None

        if m2 > m1:
            m1, m2 = m2, m1
            k1, k2 = k2, k1
            elem1, elem2 = elem2, elem1

        # now m1 >= m2, and since k1, k2 are coprime, this means that m2 = 0

        elem1 = elem1.shift(m1)

        result = BinaryExpansion._combination(1 if k1 > 0 else -1, elem1, 1 if k2 > 0 else -1, elem2)
        if result is None:
            return None

        return as_neg(result)

    @staticmethod
    def _combination(
        sign1: int, elem1: "BinaryExpansion", sign2: int, elem2: "BinaryExpansion"
    ) -> MapElement | None:

        if sign1 == sign2:
            result = elem1.try_add_binary_expansion(elem2)
            if result is not None:
                return result if sign1 == 1 else -result
        else:
            result = elem1.try_sub_binary_expansion(elem2)
            if result is not None:
                return result if sign1 == 1 else -result

            result = elem2.try_sub_binary_expansion(elem1)
            if result is not None:
                return result if sign2 == 1 else -result

        return None

    def try_add_binary_expansion(self, other: "BinaryExpansion") -> Optional["BinaryExpansion"]:

        coef1 = self.coefficients
        coef2 = other.coefficients

        n = max(len(coef1), len(coef2))
        coef1 = list(coef1) + [0] * (n - len(coef1))
        coef2 = list(coef2) + [0] * (n - len(coef2))

        carry = 0
        coefs = []
        for c1, c2 in zip(coef1, coef2):
            if carry == 1:

                if c1 == 1:
                    coefs.append(c2)
                    carry = 1
                    continue
                if c2 == 1:
                    coefs.append(c1)
                    carry = 1
                    continue

                if isinstance(c1, int) and isinstance(c2, int):
                    # must be c1 = c2 = 0
                    coefs.append(1)
                    carry = 0
                    continue

                return None

            if c1 == 0:
                coefs.append(c2)
                continue

            if c2 == 0:
                coefs.append(c1)
                continue

            if not (isinstance(c1, int) and isinstance(c2, int)):
                return None

            c = c1 + c2
            if c < 2:
                coefs.append(c)
                carry = 0
            else:
                coefs.append(c - 2)
                carry = 1

        if carry > 0:
            coefs += [1]

        return BinaryExpansion(coefs)

    def add(self, other: MapElement) -> MapElement | None:
        sign, other_elem = as_neg(other)

        if isinstance(other_elem, BinaryExpansion):

            result = BinaryExpansion._combination(1, self, sign, other_elem)
            if result is not None:
                return result

        return super().add(other)

    def radd(self, other: MapElement) -> MapElement | None:
        return self.add(other)

    def try_sub_binary_expansion(self, other: "BinaryExpansion") -> Optional["BinaryExpansion"]:
        coef1 = self.coefficients
        coef2 = other.coefficients

        n = max(len(coef1), len(coef2))
        coef1 = list(coef1) + [0] * (n - len(coef1))
        coef2 = list(coef2) + [0] * (n - len(coef2))

        carry = 0
        coefs = []
        for c1, c2 in zip(coef1, coef2):

            if carry == -1:

                if c2 == 1:
                    coefs.append(c1)
                    carry = -1
                    continue

                if c1 == c2:
                    coefs.append(1)
                    carry = -1
                    continue

                if isinstance(c1, int) and isinstance(c2, int):
                    # must be c2 = 0 and c1 = 1
                    coefs.append(0)
                    carry = 0
                    continue

                return None

            if c1 == c2:
                coefs.append(0)
                continue

            if c2 == 0:
                coefs.append(c1)
                continue

            if not (isinstance(c1, int) and isinstance(c2, int)):
                return None

            # only remaining possibility is c1 = 0 and c2 = 1

            coefs.append(1)
            carry = -1

        if carry < 0:
            return None

        return BinaryExpansion(coefs)

    def sub(self, other: MapElement) -> MapElement | None:
        sign, other_elem = as_neg(other)
        if isinstance(other_elem, BinaryExpansion):

            result = BinaryExpansion._combination(1, self, -sign, other_elem)
            if result is not None:
                return result

        return super().sub(other)

    def shift(self, k: int) -> Optional["BinaryExpansion"]:
        if k < 0:
            deg = 0
            for c in self.coefficients:
                if c != 0:
                    break
                deg += 1
            if deg + k < 0:
                return None

            return BinaryExpansion(self.coefficients[-k:])

        return BinaryExpansion([0] * k + list(self.coefficients))

    #
    #     # </editor-fold>
    #
    #     def transform_linear(self, a: int, b: int) -> tuple[int, MapElement, int]:
    #         elem, constant = self.split_constant()
    #         constant = constant.evaluate()
    #         b += a * constant
    #         if elem is None:
    #             return 0, MapElementConstant.zero, b
    #         else:
    #             return a, elem, b
    #

    def _min_max_assignment_in_range(self, low: int, high: int) -> tuple[dict[BoolVar, int], dict[BoolVar, int]]:
        # TODO: this has similar logic to transform_range method. Avoid it!
        low -= self._constant
        high -= self._constant
        assert 0 <= high and low <= self._bool_max_value[-1]

        low_dict: dict[BoolVar, int] = dict()
        high_dict: dict[BoolVar, int] = dict()

        two_power = 2 ** len(self.coefficients)
        for i in range(len(self.coefficients) - 1, -1, -1):
            c = self.coefficients[i]
            two_power //= 2
            if isinstance(c, int):
                continue

            # BoolVar
            if self._bool_max_value[i] < low:
                low_dict[c] = 1
                low -= two_power
            else:
                low_dict[c] = 0

            if two_power <= high:
                high_dict[c] = 1
                high -= two_power
            else:
                high_dict[c] = 0

        return low_dict, high_dict

    def _as_binary_range_data(
        self, condition: MapElement
    ) -> dict[Var, tuple[dict[Var, int], dict[Var, int]]] | None:

        # TODO: hate this together with me...
        value = self.evaluate()
        assert value is None  # if this binary expansion is constant, there is nothing really to do

        assert condition.has_promise(IsCondition)
        condition = condition.simplify2()

        if condition is TrueCondition:
            return next(self.promises.output_promises(of_type=InRange)).range
        if condition is FalseCondition:
            return IntervalRange.empty()
        if len(condition.vars) == 0:
            # Usually if the function doesn't depend on anything, it is a constant, and for conditions it should be
            # either True\False conditions. But some dummy conditions in test don't have variables as well, so
            # don't try to process them.
            return None

        if not (set(condition.vars) <= set(self.vars)):
            return None
        vars_order = {v: i for i, v in enumerate(self.vars)}

        if isinstance(condition, RangeCondition):
            condition = IntersectionCondition([condition])

        if not isinstance(condition, IntersectionCondition):
            return None

        # Only accept RangeConditions(BinaryExpansion) here (which include assignments), on disjoint sets of vars.
        # Also, to make my life much more simple, assumer that the vars are consecutive.
        used_vars: dict[Var, tuple[dict[Var, int], dict[Var, int]]] = {v: (dict(), dict()) for v in self.vars}
        conditions = condition.conditions
        for sub_cond in conditions:
            if not isinstance(sub_cond, RangeCondition):
                return None
            if not all(len(used_vars[v][0]) == 0 for v in sub_cond.vars):
                return None
            elem = BinaryExpansion.of(sub_cond.function)
            if elem is None:
                return None

            elem_vars_indices = [vars_order[v] for v in sub_cond.vars]
            sorted_elem_vars_indices = sorted(elem_vars_indices)
            if elem_vars_indices != list(range(sorted_elem_vars_indices[0], sorted_elem_vars_indices[-1] + 1)):
                return None

            other_range = next(elem.promises.output_promises(of_type=InRange)).range.intersection(sub_cond.range)
            if other_range is None:
                return None
            other_range = other_range.as_integral()
            if other_range is None:
                return None
            cur_low_dict, cur_high_dict = elem._min_max_assignment_in_range(other_range.low, other_range.high)
            # bin_exp = BinaryExpansion(sub_cond.vars)
            # if not (bin_exp(cur_low_dict).evaluate() <= bin_exp(point).evaluate() <= bin_exp(
            #         cur_high_dict).evaluate()):
            #     return None

            for v in reversed(elem.vars):
                if cur_low_dict[v] == cur_high_dict[v]:
                    used_vars[v] = ({v: cur_low_dict[v]}, {v: cur_low_dict[v]})
                    del cur_low_dict[v]
                    del cur_high_dict[v]
                else:
                    break

            full = True
            for v in elem.vars:
                if v not in cur_low_dict:
                    break
                if full and cur_low_dict[v] == 0 and cur_high_dict[v] == 1:
                    used_vars[v] = ({v: 0}, {v: 1})
                    del cur_low_dict[v]
                    del cur_high_dict[v]
                else:
                    full = False
                    used_vars[v] = (cur_low_dict, cur_high_dict)

        for v, value in used_vars.items():
            if len(value[0]) == 0:
                used_vars[v] = ({v: 0}, {v: 1})

        return used_vars

    def as_binary_range_containing(self, condition: MapElement, low: int, high: int) -> tuple[int, int] | None:
        """
        Tries to view the condition as a ranged condition on this BinaryExpansion (not necessarily interval range).
        If possible look for the interval containing the point, and return its upper or lower bound depending on
        the 'upper_bound' argument.
        If it could not do any of the steps above, return None.
        """
        used_vars = self._as_binary_range_data(condition)
        if used_vars is None:
            return None

        low_point, high_point = self._min_max_assignment_in_range(low - 1, high + 1)
        var_order = {v: i for i, v in enumerate(self.vars)}

        check_low = True
        check_high = True
        # TODO: possibly repeat some validations here. Unfortunately, the dict validations are unhashable to be put in a set.
        for cur_low_dict, cur_high_dict in used_vars.values():
            var_keys = sorted(cur_low_dict.keys(), key=lambda v: var_order[v])
            bin_exp = BinaryExpansion(var_keys)
            lower_bound = bin_exp(cur_low_dict).evaluate()
            upper_bound = bin_exp(cur_high_dict).evaluate()
            if check_low and not (lower_bound <= bin_exp(low_point).evaluate() <= upper_bound):
                check_low = False
            if check_high and not (lower_bound <= bin_exp(high_point).evaluate() <= upper_bound):
                check_high = False
            if not (check_low or check_high):
                return None

        for v in self.vars:
            cur_low_dict, cur_high_dict = used_vars[v]
            if len(cur_low_dict) == 1 and cur_low_dict[v] < cur_high_dict[v]:
                high_point[v] = 1
                low_point[v] = 0
                continue
            high_point.update(cur_high_dict)
            low_point.update(cur_low_dict)
            break

        return self(low_point).evaluate() if check_low else low, (self(high_point).evaluate() if check_high else high)

    # <editor-fold desc=" ------------------------ Simplifiers ------------------------ ">

    def _simplify_with_var_values2(self) -> MapElement | None:
        elem, constant = self.split_constant()
        if constant == 0:
            return None
        if elem is None:
            return constant
        # TODO: I think I want to just return elem + constant, and not use Linear
        return Linear(1, elem, constant.evaluate())

    @staticmethod
    def transform_range(range_cond: MapElement) -> MapElement | None:
        """
        Simplify an interval RangedCondition over a binary expansion.
        """
        assert isinstance(range_cond, RangeCondition)
        bin_exp = range_cond.function
        if not isinstance(bin_exp, BinaryExpansion):
            return None

        f_range = range_cond.range
        if f_range.is_empty:
            return FalseCondition
        f_range = f_range.as_integral()

        a, b = f_range.low, f_range.high
        # a = max(a, bin_exp._constant)
        # b = min(b, bin_exp._constant + bin_exp._bool_max_value[-1] + 1)

        # remove constant part
        a -= bin_exp._constant
        b -= bin_exp._constant
        coefs = [(0 if cc == 1 else cc) for cc in bin_exp.coefficients]
        if bin_exp._constant > 0:
            simplify_logger.log(f'Removed constant {bin_exp._constant}. Now simplifying: {red(f"{a}<={coefs}<={b}")}')

        # # remove zeros at the beginning
        # for i, c in enumerate(coefs):
        #     if c != 0:
        #         coefs = coefs[i:]
        #         break
        #     a = (a // 2) + (a % 2)
        #     b = (b // 2) + (b % 2)

        conditions = []

        two_power = 2 ** len(coefs)
        for i in range(len(coefs) - 1, -1, -1):
            c = coefs[i]
            two_power //= 2
            if c == 0:
                continue

            # BoolVar
            if bin_exp._bool_max_value[i] < a:
                conditions.append(c << 1)
                a -= two_power
                b -= two_power
                continue

            if b < two_power:
                conditions.append(c << 0)
                continue

            break
        else:
            i = -1

        condition = TrueCondition
        if len(conditions) == 1:
            condition = conditions[0]
        if len(conditions) > 1:
            condition = IntersectionCondition(conditions)

        if a <= 0 and bin_exp._bool_max_value[i + 1] <= b:
            return condition
        else:
            if condition is TrueCondition and (a, b) == (f_range.low, f_range.high):
                return None
            # TODO: Maybe don't simplify automatically in binary operations like &, |, etc?
            return IntersectionCondition(
                [condition, RangeCondition(BinaryExpansion(coefs[: i + 1]), IntervalRange[a, b])]
            )

    @staticmethod
    def _union_with_range_over_binary_expansion(union_elem: MapElement) -> MapElement | None:
        """
        If one of the factors in a union operation is a RangedCondition over a BinaryExpansion, try to write the union
        as another RangedCondition over the same BinaryExpansion.
        """
        assert isinstance(union_elem, UnionCondition)
        if len(union_elem.conditions) != 2:
            return None

        cond1, cond2 = union_elem.conditions
        if isinstance(cond1, RangeCondition) and isinstance(cond1.function, BinaryExpansion):
            simplify_logger.log(f"1st condition is ranged binary expansion: {red(cond1)}")
            c_range = cond1.range.as_integral()
            interval = cond1.function.as_binary_range_containing(cond2, c_range.low, c_range.high)
            if interval is not None:
                return RangeCondition(cond1.function, IntervalRange[*interval])
        if isinstance(cond2, RangeCondition) and isinstance(cond2.function, BinaryExpansion):
            simplify_logger.log(f"2nd condition is ranged binary expansion: {red(cond2)}")
            c_range = cond2.range.as_integral()
            interval = cond2.function.as_binary_range_containing(cond1, c_range.low, c_range.high)
            if interval is not None:
                return RangeCondition(cond2.function, IntervalRange[*interval])

        return None

    # TODO: there are now two simplifiers to combination to expansion
    #       After the big refactorzation, delete one of them.

    @staticmethod
    def _binary_combination_to_expansion_simplifier(bin_comb: MapElement) -> SimplifierOutput:
        """
                2 * v1 + v0     =>  Bin[v0, v1]
        """
        assert isinstance(bin_comb, BinaryCombination)

        elem1 = BinaryExpansion.of(bin_comb.elem1)
        if elem1 is None:
            return ProcessFailureReason("1st element cannot become a binary expansion")

        elem2 = BinaryExpansion.of(bin_comb.elem2)
        if elem2 is None:
            return ProcessFailureReason("2nd element cannot become a binary expansion")

        result = BinaryExpansion.linear_combination(bin_comb.c1, elem1, bin_comb.c2, elem2)
        if result is not None:
            coef, elem = result
            return Linear(coef, elem, 0)
        return None

    # </editor-fold>


RangeCondition.register_class_simplifier(BinaryExpansion.transform_range)
UnionCondition.register_class_simplifier(BinaryExpansion._union_with_range_over_binary_expansion)
BinaryCombination.register_class_simplifier(BinaryExpansion._binary_combination_to_expansion_simplifier)

# TODO: I don't think I need this anymore. Keep it here for now just in case.
# def binary_signed_addition_simplifier(var_dict: VarDict, sign: int) -> Optional[MapElement]:
#     add_vars = get_var_values((Add if sign == 1 else Sub).vars, var_dict)
#     if add_vars is None:
#         return None
#     simplify_logger.log(f'Trying to combine (Binary Expansion) {red(add_vars[0])} {"+" if sign > 0 else "-"} {red(add_vars[1])}')
#
#     linear_var1 = Linear.of(add_vars[0])
#     linear_var2 = sign*Linear.of(add_vars[1])
#
#     bin_1 = BinaryExpansion.of(linear_var1.elem)
#     bin_2 = BinaryExpansion.of(linear_var2.elem)
#     if bin_1 is None or bin_2 is None:
#         return None
#
#     result = BinaryExpansion.linear_combination(linear_var1.a, bin_1, linear_var2.a, bin_2)
#     if result is not None:
#         coef, elem = result
#         return Linear(coef, elem, linear_var1.b + linear_var2.b)
#     return None
#
# def binary_addition_simplifier(var_dict: VarDict) -> Optional[MapElement]:
#     return binary_signed_addition_simplifier(var_dict=var_dict, sign=1)
#
# def binary_subtraction_simplifier(var_dict: VarDict) -> Optional[MapElement]:
#     return binary_signed_addition_simplifier(var_dict=var_dict, sign=-1)
#
# Add.register_simplifier(binary_addition_simplifier)
#
# Sub.register_simplifier(binary_subtraction_simplifier)
