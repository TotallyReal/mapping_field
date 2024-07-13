import cmath
import random
from abc import abstractmethod
import operator
import inspect
from typing import Callable, Any, Dict, Optional, Union, List, Tuple
from field import FieldElement, ExtElement, getElement

from typing import TypeVar

Source = TypeVar('Source')
Target = TypeVar('Target')


# def _to_constant(elem):
#     if isinstance(elem, MapElementConstant):
#         return elem.get_element()
#     if isinstance(elem, int):
#         return FieldElement(elem)
#     if isinstance(elem, FieldElement):
#         return elem
#     return None


def convert_to_map(elem):

    if isinstance(elem, MapElement):
        return elem

    if isinstance(elem, int) or isinstance(elem, FieldElement):
        return MapElementConstant(elem)

    if isinstance(elem, str):
        mapElement = Var.try_get(elem)
        if mapElement is not None:
            return mapElement

        mapElement = NamedFunc.try_get(elem)
        if mapElement is not None:
            return mapElement

    return NotImplemented


def params_to_maps(f):

    def wrapper(self, element):
        value = convert_to_map(element)
        if value == NotImplemented:
            return NotImplemented
        return f(self, value)

    return wrapper


class MapElement:
    """
    The main class representing a function f: (x_1,...,x_n) -> y.

    There are two ways to apply this function:
    1. Positional: Call f(a_1, ..., a_n), where the a_i are either elements or maps.
    2. Dictionary: Call f(x_1 = a_1, ..., x_n = a_n). This dictionary access can be much more general:
            a. The variables don't have to be ordered (e.g. {x_2 = a_2, x_5 = a_5, ... }),
            b. Not all variables must appear (e.g. {x_1 = a_1, x_7 = a_7}),
            c. Extra variables can appear (e.g. {x_1 = a_1 , y_2 = b_2}),
            d. Can add assignments for functions (e.g. {g=Add, x_1=3})

    To implement this class, you must implement the function __call__ which defines the map
    """

    def __init__(self, variables: List['Var']):
        if len(variables) > len(set(variables)):
            raise Exception(f'Function must have distinct variables')
        self.vars = variables
        self.num_vars = len(variables)

    def __call__(self, *args, **kwargs) -> 'MapElement':
        var_dict = {}
        func_dict = {}
        if len(kwargs) == 0:
            if len(args) != self.num_vars:
                raise Exception(f'Function {self.name} need to get {self.num_vars} values, and instead got {len(args)}.')
            var_dict = {v: convert_to_map(value) for v, value in zip(self.vars, args)}
            args = []

        if len(args) != 0:
            raise Exception(f'When calling {self.name} use just args or just kwargs, not both.')

        for key, value in kwargs.items():
            v = Var.try_get(key)
            if v is not None:
                var_dict[v] = convert_to_map(value)
                continue

            f = NamedFunc.try_get(key)
            if f is not None:
                func_dict[f] = convert_to_map(value)
                continue

            raise Exception(f'Cannot assign new value to element which is not a variable of a named function : {key}')

        result = self._call_with_dict(var_dict, func_dict)
        return result.simplify()

    # Override when needed
    def _call_with_dict(self, var_dict: Dict['Var', 'MapElement'], func_dict: Dict['NamedFunc', 'MapElement']) -> 'MapElement':
        return self

    def evaluate(self) -> ExtElement:
        map_elem = self.simplify()
        assert isinstance(map_elem, MapElementConstant)
        return map_elem.evaluate()

    def simplify(self) -> 'MapElement':
        """
        Try to simplified the given function (e.g. 1+x*0+y -> 1+y)
        """
        return self._simplify_with_entries(self.vars)

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        """
        --------------- Override when needed ---------------
        Try to simplified the given function, given the simplified entries (which can assumed to have
        the number of entries this function needs).
        """
        return self

    def __repr__(self):
        return str(self)

    def __str__(self):
        vars_str_list = [var.name for var in self.vars]
        return self.to_string(vars_str_list)

    @abstractmethod
    def to_string(self, vars_str_list: List[str]):
        """
        --------------- Override ---------------
        Represents the function, given the string representations of its variables
        """
        pass

    # <editor-fold desc="-------------- arithmetic operations --------------">

    @params_to_maps
    def __add__(self, other):
        return Add(self, other)

    @params_to_maps
    def __radd__(self, other):
        return Add(other, self)

    @params_to_maps
    def __mul__(self, other):
        return Mult(self, other)

    @params_to_maps
    def __rmul__(self, other):
        return Mult(other, self)

    @params_to_maps
    def __sub__(self, other):
        return Sub(self, other)

    @params_to_maps
    def __rsub__(self, other):
        # After convert, other must be a MapElement
        return Sub(other,self)

    @params_to_maps
    def __truediv__(self, other):
        return Div(self, other)

    @params_to_maps
    def __rtruediv__(self, other):
        return Div(other, self)
    #
    # def __pow__(self, power: int):
    #     return UniMapping(pow_map(power), self)

    # </editor-fold>


class Var(MapElement):
    """
    A single variable. Can be thought of as the projection map on a variable, namely (x_1,...,x_i,...,x_n) -> x_n.
    The variable projected on is given by the name in the constructor.

    Cannot generate two variables with the same name. Trying to do so, will return the same variable.
    """
    _instances = {}

    @staticmethod
    def try_get(var_name: str):
        """
        Checks if there is a variable with the given name. Return it if exists, and otherwise None.
        """
        return Var._instances.get(var_name, None)

    def __new__(cls, name: str):
        if name in cls._instances:
            return cls._instances[name]

        instance = super(Var, cls).__new__(cls)
        cls._instances[name] = instance
        return instance

    def __init__(self, name: str):
        """
        Initializes the Variable. If a Variable with the given name already exists, will not create a
        second object, and instead returns the existing variable.
        """
        if hasattr(self, 'initialized'):
            return
        super().__init__([self])
        self.name = name
        self.initialized = True

    def to_string(self, vars_str_list: List[str]):
        return self.name

    def _call_with_dict(self, var_dict: Dict['Var', MapElement], func_dict: Dict['NamedFunc', MapElement]) -> MapElement:
        # Try to look both for the variable itself, and its name
        return var_dict.get(self, self)


class NamedFunc(MapElement):
    """
    A named function, which can be assigned later to another function.

    Cannot generate two functions with the same name. Trying to do so, will raise an exception.
    """
    _instances = {}

    @staticmethod
    def try_get(func_name: str):
        return NamedFunc._instances.get(func_name, None)

    def __new__(cls, func_name: str, variables: List[Var]):
        if func_name in cls._instances:
            raise Exception(f'Cannot create two functions with the same name {func_name}')

        instance = super(NamedFunc, cls).__new__(cls)
        cls._instances[func_name] = instance
        return instance

    def __init__(self, func_name: str, variables: List[Var]):
        self.name = func_name
        self.vars = variables

    def _call_with_dict(self, var_dict: Dict['Var', MapElement], func_dict: Dict['NamedFunc', MapElement]) -> MapElement:
        return func_dict.get(self, self)

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        return CompositionFunction(self, simplified_entries)

    def to_string(self, vars_str_list: List[str]):
        vars_str = ','.join(vars_str_list)
        return f'{self.name}({vars_str})'


class Func:

    def __init__(self, name: str):
        self.name = name
        self.assigned = None

    def __call__(self, *variables) -> NamedFunc:
        if self.assigned is not None:
            raise Exception(f'The name {self.name} was already assigned to a function')

        # transform variables to Var
        actual_vars = []
        for v in variables:
            if isinstance(v, Var):
                actual_vars.append(v)
                continue
            if isinstance(v, str):
                actual_vars.append(Var(v))
                continue
            raise Exception(f'Could not define the function {self.name}: Variable {v} is not well defined.')

        self.assigned = CompositionFunction(NamedFunc(self.name, actual_vars), actual_vars)
        return self.assigned


class CompositionFunction(MapElement):

    def __init__(self, function: MapElement, entries: List[MapElement]):
        seen = set()
        variables = []

        for entry in entries:
            variables += [v for v in entry.vars if v not in seen]
            seen.update(entry.vars)

        super().__init__(variables)
        self.function = function
        self.entries = entries

    def _call_with_dict(self, var_dict: Dict['Var', 'MapElement'], func_dict: Dict['NamedFunc', 'MapElement']) -> 'MapElement':
        eval_function = self.function._call_with_dict({}, func_dict)
        eval_entries = [entry._call_with_dict(var_dict, func_dict) for entry in self.entries]

        return CompositionFunction(function=eval_function, entries=eval_entries)

    #
    # def __str__(self):
    #     vars_str_list = [var.name for var in self.vars]
    #     return self.to_string(vars_str_list)

    def to_string(self, vars_str_list: List[str]):
        entries_str_list = [entry.to_string(vars_str_list) for entry in self.entries]
        return self.function.to_string(entries_str_list)

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        simplified_entries = [entry._simplify_with_entries(simplified_entries) for entry in self.entries]
        return self.function._simplify_with_entries(simplified_entries)


class MapElementConstant(MapElement):
    """
    Used for constant maps, and for casting elements into maps.
    """

    def __init__(self, elem: ExtElement):
        super().__init__([])
        self.elem = elem

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        return self

    # def __call__(self, variables: Dict) -> MapElement:
    #     return self

    # def evaluate(self) -> ExtElement:
    #     return self.elem

    # def __add__(self, other):
    #     if isinstance(other, MapElementConstant):
    #         return MapElementConstant(self.evaluate() + other.evaluate())
    #     return super().__add__(other)
    #
    # def __sub__(self, other):
    #     if isinstance(other, MapElementConstant):
    #         return MapElementConstant(self.evaluate() - other.evaluate())
    #     return super().__sub__(other)
    #
    # def __mul__(self, other):
    #     if isinstance(other, MapElementConstant):
    #         return MapElementConstant(self.evaluate() * other.evaluate())
    #     return super().__mul__(other)
    #
    # def __truediv__(self, other):
    #     if isinstance(other, MapElementConstant):
    #         return MapElementConstant(self.evaluate() / other.evaluate())
    #     return super().__truediv__(other)

    def to_string(self, vars_str_list: List[str]):
        return str(self.elem)


class MapElementFromFunction(MapElement):

    def __init__(self, name: str, function: Callable[[List[ExtElement]], ExtElement]):
        self.name = name
        self.function = function
        self.num_parameters = len(inspect.signature(function).parameters)
        variables = [Var(f'X_{name}_{i}') for i in range(self.num_parameters)]
        # TODO: Maybe use the names of the variables of the original function
        super().__init__(variables)

    def to_string(self, entries: List[str]):
        entries_str = ','.join(entries)
        return f'{self.name}({entries_str})'

    def _call_with_dict(self, var_dict: Dict['Var', 'MapElement'],
                        func_dict: Dict['NamedFunc', 'MapElement']) -> 'MapElement':
        eval_entries = [var._call_with_dict(var_dict, func_dict) for var in self.vars]

        return CompositionFunction(function=self, entries=eval_entries)

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        if any(not isinstance(entry, MapElementConstant) for entry in simplified_entries):
            return self._simplify_partial_constant(simplified_entries)

        result = self.function(*[entry.elem for entry in simplified_entries])
        return MapElementConstant(result)

    def _simplify_partial_constant(self, simplified_entries: List['MapElement']) -> 'MapElement':
        return CompositionFunction(self, simplified_entries)


class _Negative(MapElementFromFunction):

    def __init__(self):
        super().__init__('Neg', lambda a: -a)

    def to_string(self, entries: List[str]):
        return f'(-{entries[0]})'


Neg = _Negative()


class _Add(MapElementFromFunction):

    def __init__(self):
        super().__init__('Add', lambda a, b: a + b)

    def _simplify_partial_constant(self, entries: List['MapElement']) -> 'MapElement':
        if isinstance(entries[0], MapElementConstant) and (entries[0].elem == 0):
            return entries[1]
        if isinstance(entries[1], MapElementConstant) and (entries[1].elem == 0):
            return entries[0]
        # TODO: automatic equality with constants
        return super()._simplify_partial_constant(entries)

    def to_string(self, entries: List[str]):
        return f'({entries[0]}+{entries[1]})'


Add = _Add()


class _Sub(MapElementFromFunction):

    def __init__(self):
        super().__init__('Sub', lambda a, b: a + b)

    def _simplify_partial_constant(self, entries: List['MapElement']) -> 'MapElement':
        if isinstance(entries[0], MapElementConstant) and (entries[0].elem == 0):
            return Neg(entries[1])
        if isinstance(entries[1], MapElementConstant) and (entries[1].elem == 0):
            return entries[0]
        # TODO: automatic equality with constants
        return super()._simplify_partial_constant(entries)

    def to_string(self, entries: List[str]):
        return f'({entries[0]}-{entries[1]})'


Sub = _Sub()


class _Mult(MapElementFromFunction):

    def __init__(self):
        super().__init__('Mult', lambda a, b: a+b)

    def _simplify_partial_constant(self, entries: List['MapElement']) -> 'MapElement':
        if isinstance(entries[0], MapElementConstant) and (entries[0].elem == 0):
            return entries[0]
        if isinstance(entries[0], MapElementConstant) and (entries[0].elem == 1):
            return entries[1]

        if isinstance(entries[1], MapElementConstant) and (entries[1].elem == 0):
            return entries[1]
        if isinstance(entries[1], MapElementConstant) and (entries[1].elem == 1):
            return entries[0]

        return super()._simplify_partial_constant(entries)

    def to_string(self, entries: List[str]):
        return f'({entries[0]}*{entries[1]})'


Mult = _Mult()


class _Div(MapElementFromFunction):

    def __init__(self):
        super().__init__('Div', lambda a, b: a+b)

    def _simplify_partial_constant(self, entries: List['MapElement']) -> 'MapElement':
        if isinstance(entries[1], MapElementConstant):
            if entries[1].elem == 0:
                raise Exception('Cannot divide by zero')
            if entries[1].elem == 1:
                return entries[0]

        if isinstance(entries[0], MapElementConstant) and (entries[0].elem == 0):
            return entries[0]

        return super()._simplify_partial_constant(entries)

    def to_string(self, entries: List[str]):
        return f'({entries[0]}/{entries[1]})'


Div = _Div()

#
#
#
#
# SimpleUniMap = Callable[[ExtElement], ExtElement]
#
#
# def pow_map(power: int):
#     def specific_pow_map(elem: ExtElement):
#         return elem**power
#     return specific_pow_map
#
#
# class UniMapping(MapElement):
#
#     _counter = 0
#
#     def __init__(self, uni_map: Union[str, SimpleUniMap], inside: Optional[MapElement] = None):
#         self.uni_map = uni_map
#         if inside is None:
#             self.inside: MapElement = Var(f'__UniMappingVar_{UniMapping._counter}__')
#             UniMapping._counter += 1
#         else:
#             self.inside = inside
#
#     def __call__(self, variables) -> MapElement:
#         if isinstance(variables, dict):
#             uni_map = self.uni_map
#             if isinstance(self.uni_map, str):
#                 uni_map = variables.get(self.uni_map, uni_map)
#             inside = self.inside(variables)
#             return UniMapping(uni_map, inside)
#         if isinstance(self.inside, Var):
#             wrapped = convert_to_map(variables)
#             if isinstance(wrapped, MapElement):
#                 return UniMapping(self.uni_map, wrapped)
#         return NotImplemented
#
#     def simplify(self) -> MapElement:
#         inside = self.inside.simplify()
#         if isinstance(inside, MapElementConstant) and not isinstance(self.uni_map, str):
#             return MapElementConstant(self.uni_map(inside.evaluate()))
#         return UniMapping(self.uni_map, inside)
#
#     def __str__(self):
#         if isinstance(self.uni_map, str):
#             return f'{self.uni_map}({self.inside})'
#         else:
#             return f'{self.uni_map.__name__}({self.inside})'
#
#
#
#
# op_sign = {operator.add: '+', operator.sub: '-', operator.mul: '*', operator.truediv: '/'}
#
#
# class _MapElementOp(MapElement):
#     """
#     The extension of a binary operation on elements to a binary operation on functions
#     """
#
#     def __init__(
#             self, map_elem_1: MapElement, map_elem_2: MapElement,
#             op: Callable[[Any, Any], Any]):
#         self.map_elem_1 = map_elem_1
#         self.map_elem_2 = map_elem_2
#         self.op = op
#
#     def simplify(self):
#         self.map_elem_1 = self.map_elem_1.simplify()
#         self.map_elem_2 = self.map_elem_2.simplify()
#         return self.op(self.map_elem_1, self.map_elem_2)
#
#     def __call__(self, variables: Dict) -> MapElement:
#         return self.op(self.map_elem_1(variables), self.map_elem_2(variables))
#
#     def __str__(self):
#         return f'{self.map_elem_1} {op_sign[self.op]} {self.map_elem_2}'
#
# AddMap = _MapElementOp
#
#
# class ElemFunctionExtension(MapElement):
#
#     def __init__(self, name: str, function: Callable[[List[ExtElement]], ExtElement]):
#         self.name = name
#         self.function = function




