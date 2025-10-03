from typing import List, Union, Optional

from mapping_field import Var, MapElement, MapElementConstant, ExtElement


class BoolVar(Var):

    def __new__(cls, var_name: str):
        return super(BoolVar, cls).__new__(cls, var_name)

    def __init__(self, var_name: str):
        super().__init__(var_name)


class BinaryExpansion(MapElement):

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
        super().__init__([c for c in coefficients if isinstance(c, BoolVar)])
        self.coefficients = tuple([
            c if (isinstance(c, BoolVar) or c == 0 or c == 1) else 0
            # TODO: maybe remove the 1 as coefficient?
            for c in coefficients])

    def to_string(self, vars_str_list: List[str]):
        # TODO: use the vars_str_list
        vars_str = ', '.join([str(v) for v in self.coefficients])
        return f'[{vars_str}]'

    def evaluate(self) -> ExtElement:
        result = 0
        two_power = 1
        for c in self.coefficients:
            assert isinstance(c, int) # TODO: Change to return None?
            result += two_power * c
            two_power *= 2
        return result

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
                if not (isinstance(c1, int) and isinstance(c2, int)):
                    return None
                c = carry + c1 + c2
                if c < 2:
                    coefs.append(c)
                    carry = 0
                else:
                    coefs.append(c-2)
                    carry = 1
                continue

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

        result = None

        if isinstance(other, BinaryExpansion):
            result = self.try_add_binary_expansion(other)

        return super().__add__(other) if result is None else result

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
                if not (isinstance(c1, int) and isinstance(c2, int)):
                    return None
                c = carry + c1 - c2
                if c >= 0:
                    coefs.append(c)
                    carry = 0
                else:
                    coefs.append(c + 2)
                    carry = -1
                continue

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

        result = None

        if isinstance(other, BinaryExpansion):
            result = self.try_sub_binary_expansion(other)

        return super().__sub__(other) if result is None else result

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