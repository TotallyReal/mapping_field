import math

from typing import Union

from mapping_field.math_utils import gcd

ExtElement = Union["FieldElement", int]


def get_int_value(element: ExtElement) -> int:
    # returns the integer value of a field element in the range [0,_p), or the element itself
    # if it is an integer. Otherwise, returns 1 (should change to NotImplemented?)
    if isinstance(element, FieldElement):
        return element.n
    if isinstance(element, int):
        return element
    return NotImplemented


def to_field_element(element: ExtElement):
    if isinstance(element, FieldElement):
        return element
    if isinstance(element, int):
        return FieldElement(element)


def _convert_param_to_int(f):
    def wrapper(self, element: ExtElement):
        value = get_int_value(element)
        if value == NotImplemented:
            return NotImplemented
        return f(self, value)

    return wrapper


class FieldElement:

    _p = 19

    def __init__(self, n: int):
        self.n = get_int_value(n) % FieldElement._p

    # The @element parameter in the following can be either FieldElement or integer
    # Any other type will return NotImplemented, so it can try to use the same operator on
    # the element parameter (e.g. element.__add__(self) )
    @_convert_param_to_int
    def __eq__(self, element: ExtElement):
        return self.n == element % FieldElement._p

    def __req__(self, element):
        return self == element

    # ----------------------- arithmetics -------------------------

    @_convert_param_to_int
    def __add__(self, element: ExtElement) -> "FieldElement":
        return FieldElement(self.n + element)

    @_convert_param_to_int
    def __radd__(self, element: ExtElement) -> "FieldElement":
        return FieldElement(self.n + element)

    @_convert_param_to_int
    def __sub__(self, element: ExtElement) -> "FieldElement":
        return FieldElement(self.n - element)

    @_convert_param_to_int
    def __rsub__(self, element: ExtElement) -> "FieldElement":
        return FieldElement(element - self.n)

    @_convert_param_to_int
    def __mul__(self, element: ExtElement) -> "FieldElement":
        return FieldElement(self.n * element)

    @_convert_param_to_int
    def __rmul__(self, element: ExtElement) -> "FieldElement":
        return FieldElement(self.n * element)

    @_convert_param_to_int
    def __truediv__(self, element: ExtElement) -> "FieldElement":
        return self * FieldElement(element).inv()

    @_convert_param_to_int
    def __rtruediv__(self, element: ExtElement) -> "FieldElement":
        return element * self.inv()

    def __pow__(self, power: int):
        if self == 0:
            return self
        # if batch_pow doesn't do any multiplication, it will just return 1 as a number
        if power == 0:
            return FieldElement(1)
        if power > 0:
            base = self
        else:
            base = self.inv()
            power = -power
        return batch_pow(base, [power])[0]
        # return FieldElement(pow(self.n, power, self._p))

    def __abs__(self):
        return self

    # -------------------- extra arithmetics -----------------------

    def inv(self):
        if self.n == 0:
            raise ZeroDivisionError
        _, m, _ = gcd(self.n, self._p)
        return FieldElement(m)

    # ------------------------------------------------------------------

    def __str__(self):
        return str(self.n)

    def __repr__(self):
        return str(self)


def batch_pow(x, powers: list[int]):
    """
    Compute simultanously several powers of the same number x.
    """
    max_power = max(powers)
    results = [1 for _ in powers]
    if max_power == 0:
        return results
    n_bits = int(math.log2(max_power)) + 1
    for i in range(n_bits):
        mask = 1 << i
        for j in range(len(powers)):
            if powers[j] & mask:
                results[j] *= x
        x *= x
    return results


def batch_inv(elements: list[ExtElement]):
    """
    Returns a list of inverses of the given elements.
    """
    # If L=[a_1, ..., a_n], create the list [1,a_1,a_1*a_2,..., a_1*a_2*...*a_n]
    mults = [FieldElement(1)]
    for elem in elements:
        mults.append(mults[-1] * elem)
    # If P_i = a_1*...*a_i (with P_0=1), then given 1/P_n we can find (using multiplication):
    #   (1) 1/(a_n) = P_(n-1)* 1/P_n, and
    #   (2) 1/P_(n-1) = a_n* 1/P_n
    # Now use induction.
    inverse = mults[-1].inv()
    res = [0 for _ in elements]
    for i in reversed(range(len(elements))):
        res[i] = mults[i - 1] * inverse
        inverse *= elements[i]
    return res
