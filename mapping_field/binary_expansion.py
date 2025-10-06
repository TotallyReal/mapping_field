from typing import List, Union, Optional, Tuple

from mapping_field.arithmetics import as_neg
from mapping_field import CompositionFunction
from mapping_field.mapping_field import Var, MapElement, MapElementConstant, ExtElement, VarDict, FuncDict
from mapping_field.arithmetics import Mult
from mapping_field.conditions import Range, Condition, FalseCondition, TrueCondition
from mapping_field.ranged_condition import AssignmentCondition, RangeCondition, RangeTransformer
from mapping_field.linear import LinearTransformer


class BoolVar(Var, RangeTransformer):

    def __new__(cls, var_name: str):
        return super(BoolVar, cls).__new__(cls, var_name)

    def __init__(self, var_name: str):
        super().__init__(var_name)

    def transform_range(self, range_values: Range) -> Optional[Condition]:
        a, b = range_values
        if 2 <= a or b <= 0 or b <= a:
            return FalseCondition

        a = max(a, 0)
        b = min(b, 2)

        if a == 0 and b == 2:
            return TrueCondition
        return AssignmentCondition({self: 1 if a == 1 else 0})

def _two_power(k):
    k = abs(k)
    m = 0
    while k%2==0:
        k /= 2
        m += 1
    return m

class BinaryExpansion(MapElement, RangeTransformer, LinearTransformer):

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
        assert all((isinstance(c, BoolVar) or c == 0 or c == 1) for c in coefficients)
        super().__init__([c for c in coefficients if isinstance(c, BoolVar)])
        self.coefficients = tuple([
            c if (isinstance(c, BoolVar) or c == 0 or c == 1) else 0
            # TODO: maybe remove the 1 as coefficient?
            for c in coefficients])

        self._constant = 0                  # The value of this map element without the bool variables
        self._bool_max_value = [0]          # max value of only bool variables, up to position i-1

        two_power = 1
        for i, c in enumerate(self.coefficients):
            if isinstance(c, int):
                self._constant += two_power * c
                self._bool_max_value.append(self._bool_max_value[-1])
            else:
                self._bool_max_value.append(self._bool_max_value[-1] + two_power)
            two_power *= 2

    def to_string(self, vars_str_list: List[str]):
        # TODO: use the vars_str_list
        vars_str = ', '.join([str(v) for v in self.coefficients])
        return f'[{vars_str}]'

    def evaluate(self) -> ExtElement:
        assert self._bool_max_value[-1] == 0  # TODO: Change to return None?
        return self._constant

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

        try:
            value2 = other.evaluate() if isinstance(other, MapElement) else other
            value1 = self.evaluate()
            return value1 == value2
        except:
            pass


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

    # <editor-fold desc=" ------------------------ Arithmetic ------------------------">

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

    def __add__(self, other):
        sign, other_elem = as_neg(other)

        if isinstance(other_elem, BinaryExpansion):

            result = BinaryExpansion._combination(1, self, sign, other_elem)
            if result is not None:
                return result

        return super().__add__(other)

    def __radd__(self, other):
        return self + other

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

    def __sub__(self, other):
        sign, other_elem = as_neg(other)
        if isinstance(other, BinaryExpansion):

            result = BinaryExpansion._combination(1, self, -sign, other_elem)
            if result is not None:
                return result

        return super().__sub__(other)

        # result = None
        #
        # if isinstance(other, BinaryExpansion):
        #     result = self.try_sub_binary_expansion(other)
        #
        # if isinstance(other, CompositionFunction) and other.function == Neg and other.entries[0] == BinaryExpansion:
        #     result = self.try_add_binary_expansion(other.entries[0])
        #
        # return super().__sub__(other) if result is None else result

    def __rsub__(self, other):
        sign, other_elem = as_neg(other)
        if isinstance(other, BinaryExpansion):

            result = BinaryExpansion._combination(sign, other_elem, -1, self)
            if result is not None:
                return result

        return super().__rsub__(other)

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

    def __mul__(self, other):
        try:
            other = other.evaluate() if isinstance(other, MapElement) else other
            if not isinstance(other, int):
                return super().__mul__(other)

            if other == 0:
                return MapElementConstant(0)

            n = abs(other)

            k = 0
            while n % 2 == 0:
                k += 1
                n //= 2
            if other < 0:
                n *= -1

            elem = self
            if k > 0:
                elem = elem.shift(k)

            if n == 1:
                return elem
            if n == -1:
                return -elem
            return Mult(n, elem)

        except:
            return super().__mul__(other)

    def __rmul__(self, other):
        return self * other

    # </editor-fold>

    def transform_linear(self, a: int, b: int) -> Tuple[int, MapElement, int]:
        elem, constant = self.split_constant()
        constant = constant.evaluate()
        b += a * constant
        if elem is None:
            return 0, MapElementConstant(0), b

        k = _two_power(a)
        a //= 2**k
        return a, elem.shift(k), b

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
        # for c, cur_max_value in reversed(list(zip(coefs, self._bool_max_value))):
            two_power //= 2
            if c == 0:
                continue

            # BoolVar
            if self._bool_max_value[i] < a:
                condition *= AssignmentCondition({c: 1})
                a -= two_power
                b -= two_power
                continue

            if b <= two_power:
                condition *= AssignmentCondition({c: 0})
                continue

            break

        if a <= 0 and self._bool_max_value[i+1] <= b:
            return condition
        else:
            return condition * RangeCondition(BinaryExpansion(coefs[:i+1]), (a, b), simplified=True)
