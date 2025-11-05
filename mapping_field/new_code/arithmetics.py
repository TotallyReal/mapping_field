from mapping_field.log_utils.tree_loggers import TreeLogger
from mapping_field.new_code.mapping_field import (MapElement, MapElementFromFunction, MapElementConstant,
                                                  CompositionFunction,
                                                  convert_to_map, VarDict, ExtElement)
from mapping_field.serializable import DefaultSerializable
from typing import List, Tuple, Optional, Callable

simplify_logger = TreeLogger(__name__)

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

class _ArithmeticMapFromFunction(MapElementFromFunction, DefaultSerializable):
    # Create a singleton for each arithmetic function

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_ArithmeticMapFromFunction, cls).__new__(cls)
        return cls._instance

    def __init__(self, name: str, function: Callable[[List[ExtElement]], ExtElement]):
        if hasattr(self, '_initialized'):
            return
        super().__init__(name, function)
        self._initialized = True
        self._simplified = True

    @classmethod
    def try_get_entries(cls, elem: MapElement) -> Optional[Tuple[MapElement, MapElement]]:
        if not isinstance(elem, CompositionFunction):
            return None

        if elem.function is not cls._instance:
            return None

        return tuple(elem.entries)

class _Negative(_ArithmeticMapFromFunction):

    # TODO: consider transform constant(-1) into -constant(1)
    def __init__(self):
        super().__init__('Neg', lambda a: -a)

    def to_string(self, entries: List[str]):
        return f'(-{entries[0]})'

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v,v) for v in self.vars]

        if not isinstance(entries[0], CompositionFunction):
            return super()._simplify_with_var_values2(var_dict)
        function = entries[0].function
        comp_entries = entries[0].entries
        if function == Neg:
            return comp_entries[0]
        if function == Sub:
            return Sub(comp_entries[1], comp_entries[0])

        return super()._simplify_with_var_values2(var_dict)


def as_neg(map_elem: MapElement) -> Tuple[int, MapElement]:
    if isinstance(map_elem, CompositionFunction) and map_elem.function == Neg:
        return -1, map_elem.entries[0]

    return 1, map_elem


class _Add(_ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__('Add', lambda a, b: a + b)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v,v) for v in self.vars]

        if entries[0].evaluate() == 0:
            return entries[1]
        if entries[1].evaluate() == 0:
            return entries[0]

        sign0, map0 = as_neg(entries[0])
        sign1, map1 = as_neg(entries[1])
        if sign0 == -1 and sign1 == -1:
            return (-(map0+map1)).simplify2()

        if sign0 == 1 and sign1 == -1:
            # Remark: I would like to return map0 - map1, however, if any MapElement subclass defines
            #         __sub__(self, other) as self + (-other), where (-other) uses the default Neg function,
            #         this will cause an infinite loop.
            return Sub(map0, map1).simplify2()
        if sign0 == -1 and sign1 == 1:
            return Sub(map1, map0).simplify2()

        # sign0 == sign1 == 1
        return super()._simplify_with_var_values2(var_dict)

    def to_string(self, entries: List[str]):
        return f'({entries[0]}+{entries[1]})'


class _Sub(_ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__('Sub', lambda a, b: a - b)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v,v) for v in self.vars]

        if entries[0].evaluate() == 0:
            return Neg(entries[1]).simplify2()
        if entries[1].evaluate() == 0:
            return entries[0]
        if (entries[0] is entries[1]):
            # TODO:
            #   I do not use entries[0] == entries[1], because some places might use the definition for x == y
            #   as  x - y == 0. Consider adding 'equality' function that forbids this definition
            return MapElementConstant.zero

        sign0, map0 = as_neg(entries[0])
        sign1, map1 = as_neg(entries[1])

        if sign0 == -1 and sign1 == -1:
            return Sub(map1, map0).simplify2()
        if sign0 == 1 and sign1 == -1:
            return Add(map0, map1).simplify2()
        if sign0 == -1 and sign1 == 1:
            return (-Add(map1, map0)).simplify2()

        # sign0 == sign1 == 1
        return super()._simplify_with_var_values2(var_dict)

    def to_string(self, entries: List[str]):
        return f'({entries[0]}-{entries[1]})'

def _as_scalar_mult(map_elem: MapElement) -> Tuple[int, MapElement]:
    value = map_elem.evaluate()
    if value is not None:
        return value, MapElementConstant.one
    if isinstance(map_elem, CompositionFunction) and map_elem.function == Mult:
        a, b = map_elem.entries
        a_value = a.evaluate()
        b_value = b.evaluate()
        if a_value is not None:
            if b_value is not None:
                return a_value * b_value, MapElementConstant.one
            return a_value, b
        if b_value is not None:
            return b_value, a
    return 1, map_elem

# TODO: consider creating a LinearCombination class?
#       also, make this function recursive.
def _as_combination(map_elem: MapElement) -> Tuple[int, MapElement, int, MapElement]:
    if isinstance(map_elem, MapElementConstant):
        return map_elem.evaluate(), MapElementConstant.one, 0, MapElementConstant.zero

    if not isinstance(map_elem, CompositionFunction):
        return 1, map_elem, 0, MapElementConstant.zero

    function = map_elem.function

    if function is Neg:
        c0, elem0, c1, elem1 = _as_combination(map_elem.entries[0])
        return -c0, elem0, -c1, elem1

    if function is Sub:
        c0, elem0 = _as_scalar_mult(map_elem.entries[0])
        c1, elem1 = _as_scalar_mult(map_elem.entries[1])
        if c0 == 0 or elem0 is MapElementConstant.one:
            return -c1, elem1, c0, elem0
        return c0, elem0, -c1, elem1

    if function is Add:
        c0, elem0 = _as_scalar_mult(map_elem.entries[0])
        c1, elem1 = _as_scalar_mult(map_elem.entries[1])
        if c0 == 0 or elem0 is MapElementConstant.one:
            return c1, elem1, c0, elem0
        return c0, elem0, c1, elem1

    c, elem = _as_scalar_mult(map_elem)
    return c, elem, 0, MapElementConstant.zero


def _as_rational(map_elem: MapElement) -> (int, MapElement, MapElement):
    """
    :return: sign, numerator, denominator
    """
    if not isinstance(map_elem, CompositionFunction):
        return 1, map_elem, MapElementConstant.one

    sign = 1

    comp_map: CompositionFunction = map_elem
    if comp_map.function == Neg:
        sign = -1
        map_elem = comp_map.entries[0]
        if not isinstance(map_elem, CompositionFunction):
            return sign, map_elem, MapElementConstant.one

        comp_map: CompositionFunction = map_elem

    if comp_map.function == Div:
        return sign, comp_map.entries[0], comp_map.entries[1]

    return sign, map_elem, MapElementConstant.one


class _Mult(_ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__('Mult', lambda a, b: a * b)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v,v) for v in self.vars]

        # Multiplication by 0 and 1
        if entries[0].evaluate() == 0:
            return MapElementConstant.zero
        if entries[0].evaluate() == 1:
            return entries[1]

        if entries[1].evaluate() == 0:
            return MapElementConstant.zero
        if entries[1].evaluate() == 1:
            return entries[0]

        if entries[0].evaluate() == -1:
            return Neg(entries[1])
        if entries[1].evaluate() == -1:
            return Neg(entries[0])

        sign0, numerator0, denominator0 = _as_rational(entries[0])
        sign1, numerator1, denominator1 = _as_rational(entries[1])
        if entries[0] is numerator0 and entries[1] is numerator1:
            return super()._simplify_with_var_values2(var_dict)

        numerator = numerator0 * numerator1
        denominator = denominator0 * denominator1
        abs_value = numerator / denominator
        return abs_value.simplify2() if sign0 * sign1 == 1 else (-abs_value).simplify2()

    def to_string(self, entries: List[str]):
        return f'({entries[0]}*{entries[1]})'


class _Div(_ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__('Div', lambda a, b: a / b)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v,v) for v in self.vars]

        if entries[1] == 0:
            raise Exception('Cannot divide by zero')
        if entries[1] == 1:
            return entries[0]

        if entries[0] == 0:
            return entries[0]

        sign0, numerator0, denominator0 = _as_rational(entries[0])
        sign1, numerator1, denominator1 = _as_rational(entries[1])
        if entries[0] is numerator0 and entries[1] is numerator1:
            return super()._simplify_with_var_values2(var_dict)

        abs_value = ((numerator0 * denominator1) / (denominator0 * numerator1))
        return abs_value if sign0 * sign1 == 1 else -abs_value

    def to_string(self, entries: List[str]):
        return f'( {entries[0]}/{entries[1]} )'


# --------------------- Override arithmetic operators for MapElement ---------------------

def params_to_maps(f):

    def wrapper(self, element):
        value = convert_to_map(element)
        return NotImplemented if value is NotImplemented else f(self, value)

    return wrapper


Neg = _Negative()
MapElement.negation = Neg

Add = _Add()
MapElement.addition  = Add

Sub = _Sub()
MapElement.subtraction  = Sub

Mult = _Mult()
MapElement.multiplication  = Mult

Div = _Div()
MapElement.division  = Div

original_simplify_caller_function2 = MapElement._simplify_caller_function2
def _simplify_caller_function2_arithmetics(
        self: MapElement, function: 'MapElement', position: int, var_dict: VarDict) -> Optional['MapElement']:

    entries = [var_dict[v] for v in function.vars]

    if function is MapElement.negation:
        return self.neg()

    if function is MapElement.addition:
        return self.add(entries[1 - position])
    if function is MapElement.multiplication:
        return self.mul(entries[1 - position])

    if function is MapElement.subtraction:
        return self.sub(entries[1]) if position == 0 else self.rsub(entries[0])
    if function is MapElement.division:
        return self.div(entries[1]) if position == 0 else self.rdiv(entries[0])

    return original_simplify_caller_function2(self, function, position, var_dict)

MapElement._simplify_caller_function2 = _simplify_caller_function2_arithmetics


class BinaryCombination(MapElement):
    # TODO: Right now this is ONLY used for the simplification process inside Linear. Don't generate it for other
    #       reasons.
    #       Later, I should just make a LinearCombination element for expressions of the form sum c_i f_i

    def __init__(self, c1: int, elem1: MapElement, c2: int, elem2: MapElement):
        super().__init__(
            list(set(elem1.vars + elem2.vars))
        )
        self.c1 = c1
        self.elem1 = elem1
        self.c2 = c2
        self.elem2 = elem2

    def to_string(self, vars_str_list: List[str]):
        return f'Comb[{self.c1}*{self.elem1}+{self.c2}*{self.elem2}]'

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
        if self.c1 == 0:
            return self.c2 * self.elem2
        if self.c2 == 0:
            return self.c1 * self.elem1

        elem1 = self.elem1._simplify2(var_dict)
        elem2 = self.elem2._simplify2(var_dict)
        if elem1 is not None or elem2 is not None:
            elem1 = elem1 or self.elem1
            elem2 = elem2 or self.elem2
            return BinaryCombination(self.c1, elem1, self.c2, elem2)
        return None

# TODO: add tests
def _binary_combination_simplifier(comp_function: MapElement, var_dict: VarDict) -> Optional[MapElement]:
    assert isinstance(comp_function, CompositionFunction)
    if not comp_function.function in (Add, Sub):
        return None
    c1, elem1, c2, elem2 = _as_combination(comp_function)
    if c2 == 0:
        return None
    result = BinaryCombination(c1, elem1, c2, elem2)._simplify2()
    if result is None or isinstance(result, BinaryCombination):
        return None
    return result
# TODO: should class simplifiers be inherited?
CompositionFunction.register_class_simplifier(_binary_combination_simplifier)