from mapping_field import MapElement, MapElementFromFunction, MapElementConstant, CompositionFunction, convert_to_map
from typing import Callable, Any, Dict, Optional, Union, List, Tuple

"""
Implement arithmetics for the MapElement class.
This is done in a separate file for clarity. To avoid cyclic imports, I override the operator methods here.

When simplifying map with arithmetics I use the following rules:

1. Addition, subtraction, negation: 
    a. Two negations cancel each other:
                -(-a) => a
    b. Negation of subtraction always converge to single subtraction: 
                -(a-b) => b-a
    c. Sum of negation become negation of sum:
                (-a) + (-b) => -(a+b)
                
2. Multiplication, division, negation:
    Any composition of these operators, with be transformed to 
                (+-1) * (a_1 * ... * a_n) / (b_1 * ... * b_m)
    where the a_i, b_j are not multiplication, division or negation.


"""

# --------------------- MapElements for arithmetic operator ---------------------


class _Negative(MapElementFromFunction):

    # TODO: consider transform constant(-1) into -constant(1)
    def __init__(self):
        super().__init__('Neg', lambda a: -a)

    def to_string(self, entries: List[str]):
        return f'(-{entries[0]})'

    def _simplify_partial_constant(self, entries: List[MapElement]) -> MapElement:
        if not isinstance(entries[0], CompositionFunction):
            return super()._simplify_partial_constant(entries)
        function = entries[0].function
        comp_entries = entries[0].entries
        if function == Neg:
            return comp_entries[0]
        if function == Sub:
            return Sub(comp_entries[1], comp_entries[0])

        return super()._simplify_partial_constant(entries)


def _as_neg(map_elem: MapElement) -> (int, MapElement):
    if isinstance(map_elem, CompositionFunction):
        comp_map: CompositionFunction = map_elem
        if comp_map.function == Neg:
            return -1, comp_map.entries[0]

    return 1, map_elem


class _Add(MapElementFromFunction):

    def __init__(self):
        super().__init__('Add', lambda a, b: a + b)

    def _simplify_partial_constant(self, entries: List[MapElement]) -> MapElement:
        if entries[0] == 0:
            return entries[1]
        if entries[1] == 0:
            return entries[0]

        sign0, map0 = _as_neg(entries[0])
        sign1, map1 = _as_neg(entries[1])

        if sign0 == 1 and sign1 == 1:
            return super()._simplify_partial_constant(entries)
        if sign0 == 1 and sign1 == -1:
            return Sub(map0, map1)
        if sign0 == -1 and sign1 == 1:
            return Sub(map1, map0)

        # sign0 == sign1 == -1
        return -(map0+map1)

    def to_string(self, entries: List[str]):
        return f'({entries[0]}+{entries[1]})'


class _Sub(MapElementFromFunction):

    def __init__(self):
        super().__init__('Sub', lambda a, b: a - b)

    def _simplify_partial_constant(self, entries: List[MapElement]) -> MapElement:
        if entries[0] == 0:
            return Neg(entries[1])
        if entries[1] == 0:
            return entries[0]

        sign0, map0 = _as_neg(entries[0])
        sign1, map1 = _as_neg(entries[1])

        if sign0 == 1 and sign1 == 1:
            return super()._simplify_partial_constant(entries)
        if sign0 == 1 and sign1 == -1:
            return Add(map0, map1)
        if sign0 == -1 and sign1 == 1:
            return -Add(map1, map0)

        # sign0 == sign1 == -1
        return Sub(map1, map0)

    def to_string(self, entries: List[str]):
        return f'({entries[0]}-{entries[1]})'


def _as_rational(map_elem: MapElement) -> (int, MapElement, MapElement):
    if not isinstance(map_elem, CompositionFunction):
        return 1, map_elem, MapElementConstant(1)

    sign = 1

    comp_map: CompositionFunction = map_elem
    if comp_map.function == Neg:
        sign = -1
        map_elem = comp_map.entries[0]
        if not isinstance(map_elem, CompositionFunction):
            return sign, map_elem, MapElementConstant(1)

        comp_map: CompositionFunction = map_elem

    if comp_map.function == Div:
        return sign, comp_map.entries[0], comp_map.entries[1]

    return sign, map_elem, MapElementConstant(1)


class _Mult(MapElementFromFunction):

    def __init__(self):
        super().__init__('Mult', lambda a, b: a * b)

    def _simplify_partial_constant(self, entries: List[MapElement]) -> MapElement:
        # Multiplication by 0 and 1
        if entries[0] == 0:
            return entries[0]
        if entries[0] == 1:
            return entries[1]

        if entries[1] == 0:
            return entries[1]
        if entries[1] == 1:
            return entries[0]

        sign0, numerator0, denominator0 = _as_rational(entries[0])
        sign1, numerator1, denominator1 = _as_rational(entries[1])
        if entries[0] == numerator0 and entries[1] == numerator1:
            return super()._simplify_partial_constant(entries)

        numerator = numerator0 * numerator1
        denominator = denominator0 * denominator1
        abs_value = numerator / denominator
        return abs_value if sign0 * sign1 == 1 else -abs_value

    def to_string(self, entries: List[str]):
        return f'({entries[0]}*{entries[1]})'


class _Div(MapElementFromFunction):

    def __init__(self):
        super().__init__('Div', lambda a, b: a / b)

    def _simplify_partial_constant(self, entries: List[MapElement]) -> MapElement:
        if entries[1] == 0:
            raise Exception('Cannot divide by zero')
        if entries[1] == 1:
            return entries[0]

        if entries[0] == 0:
            return entries[0]

        sign0, numerator0, denominator0 = _as_rational(entries[0])
        sign1, numerator1, denominator1 = _as_rational(entries[1])
        if entries[0] == numerator0 and entries[1] == numerator1:
            return super()._simplify_partial_constant(entries)

        abs_value = ((numerator0 * denominator1) / (denominator0 * numerator1))
        return abs_value if sign0 * sign1 == 1 else -abs_value

    def to_string(self, entries: List[str]):
        return f'( {entries[0]}/{entries[1]} )'


# --------------------- Override arithmetic operators for MapElement ---------------------

def params_to_maps(f):

    def wrapper(self, element):
        value = convert_to_map(element)
        return NotImplemented if value == NotImplemented else f(self, value)

    return wrapper


Neg = _Negative()
MapElement.__neg__ = lambda self: Neg(self)

Add = _Add()
MapElement.__add__ = params_to_maps(lambda self, other: Add(self, other))
MapElement.__radd__ = params_to_maps(lambda self, other: Add(other, self))

Sub = _Sub()
MapElement.__sub__ = params_to_maps(lambda self, other: Sub(self, other))
MapElement.__rsub__ = params_to_maps(lambda self, other: Sub(other, self))

Mult = _Mult()
MapElement.__mul__ = params_to_maps(lambda self, other: Mult(self, other))
MapElement.__rmul__ = params_to_maps(lambda self, other: Mult(other, self))

Div = _Div()
MapElement.__truediv__ = params_to_maps(lambda self, other: Div(self, other))
MapElement.__rtruediv__ = params_to_maps(lambda self, other: Div(other, self))
