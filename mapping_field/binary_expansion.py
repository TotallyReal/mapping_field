from typing import List, Union, Optional, Tuple
import math

from mapping_field.log_utils.tree_loggers import TreeLogger, red
from mapping_field.mapping_field import get_var_values
from mapping_field.serializable import DefaultSerializable
from mapping_field.arithmetics import as_neg, Add, Sub
from mapping_field.mapping_field import Var, MapElement, MapElementConstant, ExtElement, VarDict, FuncDict
from mapping_field.conditions import Condition, FalseCondition, TrueCondition, ConditionUnion
from mapping_field.ranged_condition import SingleAssignmentCondition, RangeCondition, RangeTransformer, Range, \
    ConditionToRangeTransformer, RangeConditionSimplifier
from mapping_field.linear import LinearTransformer, Linear

logger = TreeLogger(__name__)

class IntVar(Var, RangeTransformer, DefaultSerializable):

    def __new__(cls, var_name: str):
        return super(IntVar, cls).__new__(cls, var_name)

    def __init__(self, var_name: str):
        super().__init__(var_name)

    @classmethod
    def serialization_name_conversion(cls):
        return {'var_name' : 'name'}

    def transform_range(self, range_values: Range) -> Optional[Condition]:
        a, b = range_values
        if a != float('-inf'):
            a = math.ceil(a)
        if b != float('inf'):
            b = math.ceil(b)

        if b == a + 1:
            return SingleAssignmentCondition(self, a)
        return None

class BoundedIntVar(IntVar):

    def __new__(cls, var_name: str, min_value: int, max_value: int):
        return super(IntVar, cls).__new__(cls, var_name)

    def __init__(self, var_name: str, min_value: int, max_value: int):
        super().__init__(var_name)
        self.min_value = min_value
        self.max_value = max_value

    def transform_range(self, range_values: Range) -> Optional[Condition]:
        a, b = range_values
        if a != float('-inf'):
            a = math.ceil(a)
        if b != float('inf'):
            b = math.ceil(b)

        a = max(a, self.min_value)
        b = min(b, self.max_value)

        if b == a + 1:
            return SingleAssignmentCondition(self, a)

        if a != range_values[0] or b != range_values[1]:
            return RangeCondition(self, (a, b))

        return None


class BoolVar(Var, RangeTransformer, DefaultSerializable):

    def __new__(cls, var_name: str):
        return super(BoolVar, cls).__new__(cls, var_name)

    def __init__(self, var_name: str):
        super().__init__(var_name)

    @classmethod
    def serialization_name_conversion(self):
        return {'var_name' : 'name'}

    def transform_range(self, range_values: Range) -> Optional[Condition]:
        a, b = range_values
        if 2 <= a or b <= 0 or b <= a:
            return FalseCondition

        a = max(a, 0)
        b = min(b, 2)

        if a == 0 and b == 2:
            return TrueCondition
        return SingleAssignmentCondition(self, 1 if a == 1 else 0)

def _range_sum_bools_simplifier(function: MapElement, f_range: Range) -> Optional[Condition]:
    entries = MapElement.addition.__class__.try_get_entries(function)
    if entries is None:
        return None

    def extract_bool_var(element: MapElement):
        if isinstance(element, Linear) and element.a == 1 and element.b == 0:
            element = element.elem
        if isinstance(element, BinaryExpansion) and all([v==0 for v in element.coefficients[1:]]):
            element = element.coefficients[0]
        if isinstance(element, BoolVar):
            return element
        return None

    var0 = extract_bool_var(entries[0])
    var1 = extract_bool_var(entries[1])

    if (var0 is None) or (var1 is None) or (entries[0] is entries[1]):
        return None

    a, b = f_range
    a = max(a, 0)
    b = min(b, 3)
    if b <= a:
        return FalseCondition

    if b-a == 3:
        return TrueCondition

    if b-a == 2:
        if a == 0:
            return ( (var0 << 0) | (var1 << 0) )
        if a == 1:
            return ( (var0 << 1) | (var1 << 1) )

    if b-a == 1:
        if a == 0:
            return ((var0 << 0) & (var1 << 0))
        if a == 1:
            return ((var0 << 0) & (var1 << 1)) | ((var0 << 1) & (var1 << 0))
        if a == 2:
            return ((var0 << 1) & (var1 << 1))

    # Not possible to get here, but just in case:
    return FalseCondition

RangeCondition.register_simplifier(_range_sum_bools_simplifier)

def _two_power(k):
    k = abs(k)
    m = 0
    while k%2==0:
        k /= 2
        m += 1
    return m

class BinaryExpansion(MapElement, RangeTransformer, LinearTransformer, ConditionToRangeTransformer, DefaultSerializable):

    @classmethod
    def of(cls, map_elem: MapElement) -> Optional['BinaryExpansion']:
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
            assert Var.try_get(f'{var_name}_{i}') is None

        return BinaryExpansion([BoolVar(f'{var_name}_{i}') for i in range(num_digits)])

    def __init__(self, coefficients: List[Union[int, BoolVar]]):
        """
        Represents an integer number in a binary expansion:

            sum_i coef[i] * 2^i

        """
        # For now, each bool variable can appear at most once
        self.coefficients = []
        for c in coefficients:
            if isinstance(c, MapElementConstant):
                c = c.evaluate()
            if isinstance(c, int):
                assert c == 0 or c == 1, f'can only use 0 or 1 as integer coefficients, instead for {c}'
                # TODO: maybe remove the 1 as coefficient?
                self.coefficients.append(c)
                continue
            assert isinstance(c, BoolVar), (f'Coefficients for Binary expansion can only be BoolVar, 0 or 1.'
                                            f'Instead got {c} of type {c.__class__}')
            self.coefficients.append(c)

        super().__init__([c for c in coefficients if isinstance(c, BoolVar)])

        self._constant = 0                  # The value of this map element without the bool variables
        self._bool_max_value = [0]          # max value of only bool variables, up to position i-1

        two_power = 1
        for i, cc in enumerate(self.coefficients):
            if isinstance(cc, int):
                self._constant += two_power * cc
                self._bool_max_value.append(self._bool_max_value[-1])
            else:
                self._bool_max_value.append(self._bool_max_value[-1] + two_power)
            two_power *= 2

    def to_string(self, vars_str_list: List[str]):
        indices = [i for i,v in enumerate(self.coefficients) if v!=0]
        if len(indices) == 0:
            return '0'
        if indices[-1] == 0:
            return str(self.coefficients[0])
        # TODO: use the vars_str_list
        vars_str = ', '.join([str(v) for v in self.coefficients[:1+indices[-1]]])
        return f'Bin[{vars_str}]'

    def evaluate(self) -> Optional[ExtElement]:
        return self._constant if (self._bool_max_value[-1] == 0) else None

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        coefs = [(c if isinstance(c,int) else var_dict.get(c,c)) for c in self.coefficients]
        if self.coefficients == coefs:
            return self
        return BinaryExpansion(coefs)

    def split_constant(self) -> Tuple[Optional['BinaryExpansion'], MapElementConstant]:
        constant_part = MapElementConstant(self._constant)
        if self._constant == 0:
            return (self, constant_part)

        if self._bool_max_value[-1] == 0:
            return (None, constant_part)

        coefs = [0 if isinstance(c, int) else c for c in self.coefficients]
        return (BinaryExpansion(coefs), constant_part)

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

        return (all([c1 == c2 for c1, c2 in zip(coef1, coef2)]) and
                all([c1 == 0 for c1 in coef1[n:]]) and
                all([c2 == 0 for c2 in coef2[n:]]) )

    def as_assignment(self, k: int) -> Condition:
        return RangeCondition(self, (k, k+1)).simplify()

    # <editor-fold desc=" ------------------------ Arithmetic ------------------------">

    def linear_combination(self, k1: int, k2: int, elem2: MapElement) -> Optional[Tuple[int, MapElement]]:
        if not isinstance(elem2, BinaryExpansion):
            return None

        elem1 = self
        logger.log(f'Trying to linear combine to Binary Expansion {red(k1)} * {red(elem1)} + {red(k2)} * {red(elem2)}')

        coef1 = elem1.coefficients
        non_zero1 = [i for i, v in enumerate(coef1) if v!=0]
        if k1 == 0 or len(non_zero1) == 0:
            return k2, elem2

        coef2 = elem2.coefficients
        non_zero2 = [i for i, v in enumerate(coef2) if v!=0]
        if k2 == 0 or len(non_zero2) == 0:
            return k1, elem1

        # Check if both elements are the same up to a power of 2
        if len(non_zero1) == len(non_zero2):
            diff = non_zero1[0] - non_zero2[0]
            if all( ((coef1[i1] == coef2[i2]) and (i1 - i2 == diff)) for i1,i2 in zip(non_zero1, non_zero2)):
                if diff >= 0:
                    return (2 ** diff) * k1 + k2, elem2
                else:
                    return (2 ** (-diff)) * k2 + k1, elem1

        # Under assumption, k1 and k2 are coprime

        m1 = _two_power(k1)
        m2 = _two_power(k2)
        if abs(k1) != 2**m1 or abs(k2) != 2**m2:
            return None

        elem1 = self
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
    def _combination(sign1: int, elem1: 'BinaryExpansion', sign2: int, elem2: 'BinaryExpansion') -> Optional[MapElement]:

        if sign1 == sign2:
            result = elem1.try_add_binary_expansion(elem2)
            if result is not None:
                return result if sign1==1 else -result
        else:
            result = elem1.try_sub_binary_expansion(elem2)
            if result is not None:
                return result if sign1==1 else -result

            result = elem2.try_sub_binary_expansion(elem1)
            if result is not None:
                return result if sign2==1 else -result

        return None

    def try_add_binary_expansion(self, other: 'BinaryExpansion') -> Optional['BinaryExpansion']:

        coef1 = self.coefficients
        coef2 = other.coefficients

        n = max(len(coef1), len(coef2))
        coef1 = list(coef1) + [0] * (n-len(coef1))
        coef2 = list(coef2) + [0] * (n-len(coef2))

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

                if (isinstance(c1, int) and isinstance(c2, int)):
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

    def add(self, other: MapElement) -> Optional[MapElement]:
        sign, other_elem = as_neg(other)

        if isinstance(other_elem, BinaryExpansion):

            result = BinaryExpansion._combination(1, self, sign, other_elem)
            if result is not None:
                return result

        return super().add(other)

    def radd(self, other: MapElement) -> Optional[MapElement]:
        return self.add(other)

    def try_sub_binary_expansion(self, other: 'BinaryExpansion') -> Optional['BinaryExpansion']:
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

                if (isinstance(c1, int) and isinstance(c2, int)):
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

    def sub(self, other: MapElement) -> Optional[MapElement]:
        sign, other_elem = as_neg(other)
        if isinstance(other, BinaryExpansion):

            result = BinaryExpansion._combination(1, self, -sign, other_elem)
            if result is not None:
                return result

        return super().sub(other)

    def shift(self, k: int) -> Optional['BinaryExpansion']:
        if k < 0:
            deg = 0
            for c in self.coefficients:
                if c!=0:
                    break
                deg += 1
            if deg + k < 0:
                return None

            return BinaryExpansion(self.coefficients[-k:])

        return BinaryExpansion([0] * k + list(self.coefficients))

    # </editor-fold>

    def transform_linear(self, a: int, b: int) -> Tuple[int, MapElement, int]:
        elem, constant = self.split_constant()
        constant = constant.evaluate()
        b += a * constant
        if elem is None:
            return 0, MapElementConstant.zero, b
        else:
            return a, elem, b

    def transform_range(self, range_values: Range) -> Optional[Condition]:
        a, b = range_values
        a = max(a, self._constant)
        b = min(b, self._constant + self._bool_max_value[-1] + 1)

        if b <= a:
            return FalseCondition

        # remove constant part
        a -= self._constant
        b -= self._constant
        coefs = [(0 if cc == 1 else cc) for cc in self.coefficients]

        # # remove zeros at the beginning
        # for i, c in enumerate(coefs):
        #     if c != 0:
        #         coefs = coefs[i:]
        #         break
        #     a = (a // 2) + (a % 2)
        #     b = (b // 2) + (b % 2)

        condition = TrueCondition

        two_power = 2**len(coefs)
        for i in range(len(coefs)-1,-1,-1):
            c = coefs[i]
            two_power //= 2
            if c == 0:
                continue

            # BoolVar
            if self._bool_max_value[i] < a:
                condition &= SingleAssignmentCondition(c, 1)
                a -= two_power
                b -= two_power
                continue

            if b <= two_power:
                condition &= SingleAssignmentCondition(c, 0)
                continue

            break
        else:
            i = -1

        if a <= 0 and self._bool_max_value[i+1] < b: # TODO: +1 ?
            return condition
        else:
            if condition is TrueCondition and (a, b) == range_values:
                return None
            return condition & RangeCondition(BinaryExpansion(coefs[:i+1]), (a, b), simplified=True)

    def as_range(self, condition: Condition) -> Optional[Range]:
        var_dict = SingleAssignmentCondition.as_assignment_dict(condition)
        if var_dict is None:
            return None

        if not all([isinstance(value_, int) for value_ in var_dict.values()]):
            return None

        var_dict = var_dict.copy()
        a = 0
        b = 0
        two_power = 2 ** len(self.coefficients)
        for c in reversed(self.coefficients):
            two_power //= 2
            if isinstance(c, int):
                a += c * two_power
                b += c * two_power
                continue

            if c in var_dict:
                value = var_dict[c]
                del var_dict[c]
                a += value * two_power
                b += value * two_power
                continue

            if len(var_dict) > 0:
                # TODO: split to assignment + range?
                return None
            b += two_power

        return (a,b+1)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
        elem, constant = self.split_constant()
        if constant == 0:
            return None
        if elem is None:
            return constant
        return Linear(1, elem, constant.evaluate())

def binary_signed_addition_simplifier(var_dict: VarDict, sign: int) -> Optional[MapElement]:
    add_vars = get_var_values((Add if sign == 1 else Sub).vars, var_dict)
    if add_vars is None:
        return None
    logger.log(f'Trying to combine (Binary Expansion) {red(add_vars[0])} {"+" if sign>0 else "-"} {red(add_vars[1])}')

    linear_var1 = Linear.of(add_vars[0])
    linear_var2 = sign*Linear.of(add_vars[1])

    bin_1 = BinaryExpansion.of(linear_var1.elem)
    bin_2 = BinaryExpansion.of(linear_var2.elem)
    if bin_1 is None or bin_2 is None:
        return None

    result = bin_1.linear_combination(linear_var1.a, linear_var2.a, bin_2)
    if result is not None:
        coef, elem = result
        return Linear(coef, elem, linear_var1.b + linear_var2.b)
    return None

def binary_addition_simplifier(var_dict: VarDict) -> Optional[MapElement]:
    return binary_signed_addition_simplifier(var_dict=var_dict, sign=1)

def binary_subtraction_simplifier(var_dict: VarDict) -> Optional[MapElement]:
    return binary_signed_addition_simplifier(var_dict=var_dict, sign=-1)

Add.register_simplifier(binary_addition_simplifier)

Sub.register_simplifier(binary_subtraction_simplifier)
