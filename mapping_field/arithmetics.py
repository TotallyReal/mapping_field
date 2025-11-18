from typing import Callable, Dict, List, Optional, Tuple, Union

from mapping_field.log_utils.tree_loggers import TreeLogger
from mapping_field.mapping_field import (
    CompositionFunction, ExtElement, MapElement, MapElementConstant, MapElementFromFunction, Var,
    VarDict, convert_to_map,
)
from mapping_field.processors import ProcessFailureReason
from mapping_field.serializable import DefaultSerializable

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
        if hasattr(self, "_initialized"):
            return
        super().__init__(name, function, simplified=True)
        self._initialized = True

    @classmethod
    def try_get_entries(cls, elem: MapElement) -> Optional[Tuple[MapElement, MapElement]]:
        if not isinstance(elem, CompositionFunction):
            return None

        if elem.function is not cls._instance:
            return None

        return tuple(elem.entries)


# <editor-fold desc=" ------------------- Negation ------------------- ">


class _Negative(_ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__("Neg", lambda a: -a)
        self.register_simplifier(_Negative._to_negation_simplifier)

    def to_string(self, vars_to_str: Dict[Var, str]):
        return f"(-{vars_to_str.get(self.vars[0])})"

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v, v) for v in self.vars]

        if not isinstance(entries[0], CompositionFunction):
            return super()._simplify_with_var_values2(var_dict)
        function = entries[0].function
        comp_entries = entries[0].entries
        if function == Neg:
            return comp_entries[0]
        if function == Sub:
            return Sub(comp_entries[1], comp_entries[0])

        return super()._simplify_with_var_values2(var_dict)

    @staticmethod
    def _to_negation_simplifier(var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict[v] for v in Neg.vars]
        return entries[0].neg()

Neg = _Negative()
MapElement.negation = Neg


# </editor-fold>


# <editor-fold desc=" ------------------- Addition ------------------- ">

class _Add(_ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__("Add", lambda a, b: a + b)
        self.register_simplifier(_Add._to_add_simplifier)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v, v) for v in self.vars]

        if entries[0].evaluate() == 0:
            return entries[1]
        if entries[1].evaluate() == 0:
            return entries[0]

        sign0, map0 = as_neg(entries[0])
        sign1, map1 = as_neg(entries[1])
        if sign0 == -1 and sign1 == -1:
            return (-(map0 + map1)).simplify2()

        if sign0 == 1 and sign1 == -1:
            # Remark: I would like to return map0 - map1, however, if any MapElement subclass defines
            #         __sub__(self, other) as self + (-other), where (-other) uses the default Neg function,
            #         this will cause an infinite loop.
            return Sub(map0, map1).simplify2()
        if sign0 == -1 and sign1 == 1:
            return Sub(map1, map0).simplify2()

        # sign0 == sign1 == 1
        return super()._simplify_with_var_values2(var_dict)

    def to_string(self, vars_to_str: Dict[Var, str]):
        entries = [vars_to_str.get(v, v) for v in self.vars]
        return f"({entries[0]}+{entries[1]})"

    @staticmethod
    def _to_add_simplifier(var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict[v] for v in Add.vars]
        return entries[0].add(entries[1]) or entries[1].add(entries[0])

Add = _Add()
MapElement.addition = Add

# </editor-fold>


# <editor-fold desc=" ------------------- Subtraction ------------------- ">

class _Sub(_ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__("Sub", lambda a, b: a - b)
        self.register_simplifier(_Sub._to_sub_simplifier)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v, v) for v in self.vars]

        if entries[0].evaluate() == 0:
            return Neg(entries[1]).simplify2()
        if entries[1].evaluate() == 0:
            return entries[0]
        if entries[0] is entries[1]:
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

    def to_string(self, vars_to_str: Dict[Var, str]):
        entries = [vars_to_str.get(v, v) for v in self.vars]
        return f"({entries[0]}-{entries[1]})"

    @staticmethod
    def _to_sub_simplifier(var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict[v] for v in Sub.vars]
        return entries[0].sub(entries[1]) or entries[1].rsub(entries[0])

Sub = _Sub()
MapElement.subtraction = Sub

# </editor-fold>


# <editor-fold desc=" ------------------- Multiplication ------------------- ">


class _Mult(_ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__("Mult", lambda a, b: a * b)
        self.register_simplifier(_Mult._to_mult_simplifier)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v, v) for v in self.vars]

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

    def to_string(self, vars_to_str: Dict[Var, str]):
        entries = [vars_to_str.get(v, v) for v in self.vars]
        return f"({entries[0]}*{entries[1]})"

    @staticmethod
    def _to_mult_simplifier(var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict[v] for v in Mult.vars]
        return entries[0].mul(entries[1]) or entries[1].mul(entries[0])

Mult = _Mult()
MapElement.multiplication = Mult

# </editor-fold>


# <editor-fold desc=" ------------------- Division ------------------- ">


class _Div(_ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__("Div", lambda a, b: a / b)
        self.register_simplifier(_Div._to_div_simplifier)

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v, v) for v in self.vars]

        if entries[1] == 0:
            raise Exception("Cannot divide by zero")
        if entries[1] == 1:
            return entries[0]

        if entries[0] == 0:
            return entries[0]

        sign0, numerator0, denominator0 = _as_rational(entries[0])
        sign1, numerator1, denominator1 = _as_rational(entries[1])
        if entries[0] is numerator0 and entries[1] is numerator1:
            return super()._simplify_with_var_values2(var_dict)

        abs_value = (numerator0 * denominator1) / (denominator0 * numerator1)
        return abs_value if sign0 * sign1 == 1 else -abs_value

    def to_string(self, vars_to_str: Dict[Var, str]):
        entries = [vars_to_str.get(v, v) for v in self.vars]
        return f"( {entries[0]}/{entries[1]} )"

    @staticmethod
    def _to_div_simplifier(var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict[v] for v in Div.vars]
        return entries[0].mul(entries[1]) or entries[1].mul(entries[0])

Div = _Div()
MapElement.division = Div

# </editor-fold>


# Arithmetic decompositions

def as_neg(map_elem: MapElement) -> Tuple[int, MapElement]:
    entries = map_elem.get_entries(_Negative)
    return (-1, entries[0]) if entries is not None else (1, map_elem)

def _as_scalar_mult(map_elem: MapElement) -> Tuple[int, MapElement]:
    value = map_elem.evaluate()
    if value is not None:
        return value, MapElementConstant.one

    entries = map_elem.get_entries(_Mult)
    if entries is not None:
        a, b = entries
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


class BinaryCombination(MapElement):
    # TODO: Right now this is ONLY used for the simplification process inside Linear. Don't generate it for other
    #       reasons.
    #       Later, I should just make a LinearCombination element for expressions of the form sum c_i f_i

    def __init__(self, c1: int, elem1: MapElement, c2: int, elem2: MapElement):
        super().__init__(list(set(elem1.vars + elem2.vars)))
        self.c1 = c1
        self.elem1 = elem1
        self.c2 = c2
        self.elem2 = elem2

    def to_string(self, vars_to_str: Dict[Var, str]):
        return f"Comb[{self.c1}*{self.elem1.to_string(vars_to_str)}+{self.c2}*{self.elem2.to_string(vars_to_str)}]"

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
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
def _binary_combination_simplifier(
    comp_function: MapElement, var_dict: VarDict
) -> Optional[Union[MapElement, ProcessFailureReason]]:
    assert isinstance(comp_function, CompositionFunction)
    if not comp_function.function in (Add, Sub):
        return ProcessFailureReason("Function is not Add or Sub", trivial=True)
    c1, elem1, c2, elem2 = _as_combination(comp_function)
    if c2 == 0:
        return None
    result = BinaryCombination(c1, elem1, c2, elem2)._simplify2()
    if result is None or isinstance(result, BinaryCombination):
        return None
    return result


# TODO: should class simplifiers be inherited?
CompositionFunction.register_class_simplifier(_binary_combination_simplifier)

def _binary_commutative_simplifier(
    comp_function: MapElement, var_dict: VarDict
) -> Optional[Union[MapElement, ProcessFailureReason]]:
    assert isinstance(comp_function, CompositionFunction)
    if not comp_function.function in (Add, Mult):
        return ProcessFailureReason("Only applicable to Add or Mult", trivial=True)
    entries = comp_function.entries
    if str(entries[0]) > str(entries[1]):
        # TODO: This actually changes the function itself
        commute_entries = CompositionFunction(comp_function.function, [entries[1], entries[0]])
        commute_entries.promises = commute_entries.promises.copy()
        return commute_entries
    return ProcessFailureReason("Did not need to change the order of parts", trivial=True)
CompositionFunction.register_class_simplifier(_binary_commutative_simplifier)
