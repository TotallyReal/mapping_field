from abc import abstractmethod
import collections
import inspect
from typing import Callable, Dict, List, Optional
from mapping_field.field import FieldElement, ExtElement


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
        map_element = Var.try_get(elem)
        if map_element is not None:
            return map_element

        map_element = NamedFunc.try_get(elem)
        if map_element is not None:
            return map_element

    return NotImplemented


VarDict = Dict['Var', 'MapElement']
FuncDict = Dict['NamedFunc', 'MapElement']

def get_var_values(var_list: List['Var'], var_dict: VarDict) -> Optional[List['MapElement']]:
    """
    Looks for the valuations of the given variables, and return them as a list, if at least one of them
    is not trivial. Otherwise, returns None
    """
    if len(var_dict) == 0:
        return None
    eval_entries = []
    trivial = True
    for var in var_list:
        eval_var = var_dict.get(var, var)
        eval_entries.append(eval_var)
        if eval_var != var:
            trivial = False

    return None if trivial else eval_entries


class MapElement:
    """
    The main class representing a "formula" which both has variables, and function variables.

    For example:
            PHI(x, y, f) := sin(x*y) + f(x - y)
    This function has two standard variables x, y and one function variable f

    As the standard variables are more common, for ease of use they must be set in the __init__ function.
    This order is used when calling the function in __call__. You can override it in subclasses, but better to override
    the _call_with_dict(var_dict, func_dict) method instead. (see description below)
    """

    def __init__(self, variables: List['Var']):
        """
        The 'variables' are the ordered list used when calling the function, as in f(a_1,...,a_n).
        """
        var_names = set(v.name for v in variables)
        if len(variables) > len(var_names):
            raise Exception(f'Function must have distinct variables')
        self.vars = variables
        self.num_vars = len(variables)

    def set_var_order(self, variables: List['Var']):
        """
        Reset the order of the standard variables
        """
        if len(variables) > len(set(variables)):
            raise Exception(f'Function must have distinct variables')

        if collections.Counter(variables) != collections.Counter(self.vars):
            raise Exception(f'New variables order {variables} have to be on the function\'s variables {self.vars}')

        self.vars = variables

    # <editor-fold desc=" ------------------------ String represnetation ------------------------">

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

    # </editor-fold>


    # <editor-fold desc=" ------------------------ Call and Simplify function ------------------------">

    def __call__(self, *args, **kwargs) -> 'MapElement':
        """
        There are three ways to apply this function:
        1. Positional: Call f(a_1, ..., a_n), where the a_i are either elements or maps.
        2. Keywords:   Call f(x_1 = a_1, ..., x_n = a_n, f_1 = F_1, ..., f_k = F_k)
           Provides a more general approach than the positional :
                a. The variables don't have to be ordered (e.g. {x_2 = a_2, x_5 = a_5, ... }),
                b. Not all variables must appear (e.g. {x_1 = a_1, x_7 = a_7}),
                c. Extra variables can appear (e.g. {x_1 = a_1 , y_2 = b_2}),
                d. Can add assignments for function variables (e.g. {g=Add, x_1=3})
        3. Dictionary: Call f({x_1 : a_1, ..., x_n : a_n, f_1 : F_1, ..., f_k : F_k})
           Must have a single unnamed argument. Behaviour is similar to the keyword approach,
           only you can use for keys both the variables themselves and their names.

        To implement this method in a subclass, you must implement the function _call_with_dict below.
        """
        # Extract simplify flag
        simplify = True
        if 'simplify' in kwargs:
            simplify = kwargs['simplify']
            if not isinstance(simplify, bool):
                raise Exception(f'The "simplify" flag must be a boolean, instead got {simplify}')
            del kwargs['simplify']

        if len(args) != 0 and len(kwargs) != 0:
            raise Exception(f'When calling a function use just args or just kwargs, not both.')

        var_dict = {}
        func_dict = {}
        if len(kwargs) == 0:
            if len(args) == 1 and isinstance(args[0], Dict):
                kwargs = args[0]
                args = []
            else:
                if len(args) != self.num_vars:
                    raise Exception(f'Function needs to get {self.num_vars} values, and instead got {len(args)}.')
                var_dict = {v: convert_to_map(value) for v, value in zip(self.vars, args)}

        # Split assignments into variables and functions
        for key, value in kwargs.items():

            v = key if isinstance(key, Var) else None
            if isinstance(key, str):
                v = Var.try_get(key)
            if v is not None:
                var_dict[v] = convert_to_map(value)
                continue

            f = key if isinstance(key, NamedFunc) else None
            f = NamedFunc.try_get(key)
            if f is not None:
                assigned_function = convert_to_map(value)
                if f.num_vars != assigned_function.num_vars:
                    raise Exception(
                        f'Cannot assign function {f} with {f.num_vars} variables to '
                        f'{assigned_function} with {assigned_function.num_vars} variables')
                func_dict[f] = assigned_function
                continue

            raise Exception(f'Cannot assign new value to element which is not a variable of a named function : {key}')

        result = self._call_with_dict(var_dict, func_dict)
        return result.simplify() if simplify else result

    # Override when needed
    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        """
        Apply the map with the given values of the standard and function variables
        """
        return self

    def evaluate(self) -> ExtElement:
        """
        Returns the constant this map defines. If it is not constant, raises an error.
        """
        map_elem = self.simplify()
        assert isinstance(map_elem, MapElementConstant)
        return map_elem.evaluate()

    def is_zero(self) -> bool:
        try:
            return self.evaluate() == 0
        except:
            return False

    def simplify(self) -> 'MapElement':
        """
        Try to simplify the given function (e.g. 1 + 0*x + y -> 1+y ).
        The resulting function should compute the same function as the current one, on the same standard variables,
        and the same order. The only difference is how it is computed inside python
        """
        return self._simplify_with_var_values({v:v for v in self.vars})

    def _entry_list(self, var_dict: VarDict):
        return [var_dict.get(v, v) for v in self.vars]

    # Override when needed
    def _simplify_with_var_values(self, var_dict: VarDict) -> 'MapElement':
        """
        --------------- Override when needed ---------------
        Try to simplify the given function, given assignment of the function standard variables.

        For example, the function
            (x,y) -> x*y
        cannot be simplified, but if we know that one of the entries is 0, then it can be simplified to zero,
        and if x=1, then it can be simplified to (x,y) -> y , and similarly with y=1.
        """
        return self

    # </editor-fold>

    # Overriding the following functions in the arithmetics.py file.
    # Adding them here to help the compiler know that they exist.

    # <editor-fold desc=" ------------------------ Arithmetic functions ------------------------">

    @staticmethod
    def addition(elem1: 'MapElement' ,elem2: 'MapElement') -> 'MapElement':
        """
        A default implementation, overriden in arithmetics.py
        """
        return NotImplemented

    def add(self, other: 'MapElement') -> 'MapElement':
        """
        Tries to add this element with 'other', assuming it is a MapElement.
        Override in subclass when it is not the default behaviour.
        """
        return NotImplemented

    def __add__(self, other) -> 'MapElement':
        """
        Override this method only for adding new type of objects other than:
        int, float, FieldElement and MapElement.
        """
        other = convert_to_map(other)
        if other is NotImplemented:
            return NotImplemented

        # Some absolutely default behaviour
        if self == 0:
            return other
        if other == 0:
            return self

        result = self.add(other)
        if result is not NotImplemented:
            return result

        result = other.add(self)
        if result is not NotImplemented:
            return result

        return MapElement.addition(self, other)

    def __radd__(self, other) -> 'MapElement':
        return self.__add__(other)

    def __neg__(self) -> 'MapElement':
        return NotImplemented

    def __sub__(self, other) -> 'MapElement':
        return NotImplemented

    def __rsub__(self, other) -> 'MapElement':
        return NotImplemented

    @staticmethod
    def multiplication(elem1: 'MapElement' ,elem2: 'MapElement') -> 'MapElement':
        """
        A default implementation, overriden in arithmetics.py
        """
        return NotImplemented

    def mul(self, other: 'MapElement') -> 'MapElement':
        """
        Tries to multiply this element with 'other', assuming it is a MapElement.
        Override in subclass when it is not the default behaviour.
        """
        return NotImplemented

    def __mul__(self, other) -> 'MapElement':
        """
        Override this method only for adding new type of objects other than:
        int, float, FieldElement and MapElement.
        """
        other = convert_to_map(other)
        if other is NotImplemented:
            return NotImplemented

        # Some absolutely default behaviour
        if self == 0 or other == 0:
            return MapElementConstant.zero
        if self == 1:
            return other
        if other == 1:
            return self
        # TODO: Consider processing multiplying by (-1) as taking the negative.
        #       Make sure there are no infinite loops, if neg is defined by multiplying by (-1)

        result = self.mul(other)
        if result is not NotImplemented:
            return result

        result = other.mul(self)
        if result is not NotImplemented:
            return result

        return MapElement.multiplication(self, other)

    def __rmul__(self, other) -> 'MapElement':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'MapElement':
        return NotImplemented

    def __rtruediv__(self, other) -> 'MapElement':
        return NotImplemented

    # </editor-fold>


class Var(MapElement):
    """
    A single variable. Can be thought of as the projection map on a variable, namely (x_1,...,x_i,...,x_n) -> x_i.
    The variable projected on is given by the name in the constructor.

    Cannot generate two variables with the same name. Trying to do so, will return the same variable.
    """
    _instances: Dict[str, 'Var'] = {}

    @classmethod
    def try_get(cls, var_name: str) -> Optional['Var']:
        """
        Checks if there is a variable with the given name. Return it if exists, and otherwise None.
        """
        return cls._instances.get(var_name, None)

    # TODO: Consider using __class_getitem__ for the try_get method

    @classmethod
    def clear_vars(cls):
        cls._instances = {}

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
        self.name = name
        super().__init__([self])
        self.initialized = True

    def to_string(self, vars_str_list: List[str]):
        return self.name

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        return var_dict.get(self, self)

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(('Var', self.name))


class NamedFunc(MapElement):
    """
    A named function, which can be assigned later to another function.

    Cannot generate two functions with the same name. Trying to do so, will raise an exception.
    """
    _instances: Dict[str, 'NamedFunc'] = {}

    @classmethod
    def try_get(cls, func_name: str) -> Optional['NamedFunc']:
        return cls._instances.get(func_name, None)

    @classmethod
    def clear_vars(cls):
        cls._instances = {}

    def __new__(cls, func_name: str, variables: List[Var]):
        if func_name in cls._instances:
            # TODO: Consider creating a specified exception
            raise Exception(f'Cannot create two functions with the same name {func_name}')

        instance = super(NamedFunc, cls).__new__(cls)
        cls._instances[func_name] = instance
        return instance

    def __init__(self, func_name: str, variables: List[Var]):
        super().__init__(variables)
        self.name = func_name

    def to_string(self, vars_str_list: List[str]):
        vars_str = ','.join(vars_str_list)
        return f'{self.name}({vars_str})'

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:

        func = func_dict.get(self, self)
        if func != self:
            return func._call_with_dict(var_dict, {})

        eval_entries = get_var_values(self.vars, var_dict)
        return self if eval_entries is None else CompositionFunction(function=self, entries=eval_entries)

    def _simplify_with_var_values(self, var_dict: VarDict) -> 'MapElement':
        eval_entries = get_var_values(self.vars, var_dict)
        return self if eval_entries is None else CompositionFunction(function=self, entries=eval_entries)


class Func:
    """
    A helper class used to create a Named Function map.
    Instead of
        NamedFunc('f',[X_1, ..., X_n])
    you can use
        Func('f')(X_1,...,X_n)
    """

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

        # TODO: why composition
        self.assigned = NamedFunc(self.name, actual_vars)
        return self.assigned


class CompositionFunction(MapElement):

    def __init__(self, function: MapElement, entries: List[MapElement]):
        """
        The composition of the given function with the entries of that function.
        The number of entries should be the number of standard variables of the function, and in
        the same order.
        """
        seen = set()
        variables = []

        for entry in entries:
            variables += [v for v in entry.vars if v not in seen]
            seen.update(entry.vars)

        super().__init__(variables)
        if isinstance(function, CompositionFunction):
            top_function = function.function
            var_dict = {var: entry for var, entry in zip(function.vars, entries)}
            top_entries = [entry._call_with_dict(var_dict, {}) for entry in function.entries]
            self.function = top_function
            self.entries = top_entries
        else:
            self.function = function
            self.entries = entries

    def to_string(self, vars_str_list: List[str]):
        # Compute the str representation for each entry, by supplying it the str
        # representations of its variables
        var_str_dict = {var: var_str for var, var_str in zip(self.vars, vars_str_list)}
        entries_str_list = [
            entry.to_string([var_str_dict[var] for var in entry.vars])
            for entry in self.entries]
        return self.function.to_string(entries_str_list)

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        if len(var_dict) == 0 and len(func_dict) == 0:
            return self
        eval_function = self.function._call_with_dict({}, func_dict)
        eval_entries = [entry._call_with_dict(var_dict, func_dict) for entry in self.entries]

        return CompositionFunction(function=eval_function, entries=eval_entries)

    def _simplify_with_var_values(self, var_dict: VarDict) -> 'MapElement':
        # Compute the simplified entries, by supplying each with the simplified version
        # of its variables
        simplified_entries = { v : entry._simplify_with_var_values(var_dict)
                               for v, entry in zip(self.function.vars, self.entries)}
        return self.function._simplify_with_var_values(simplified_entries)


class MapElementConstant(MapElement):
    """
    Used for constant maps, and for casting elements into maps.
    """

    def __init__(self, elem: ExtElement):
        super().__init__([])
        self.elem = elem

    def to_string(self, vars_str_list: List[str]):
        return str(self.elem)

    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, FieldElement):
            return self.elem == other
        if isinstance(other, MapElementConstant):
            return self.elem == other.elem
        return super().__eq__(other)

    def evaluate(self) -> ExtElement:
        return self.elem

MapElementConstant.zero = MapElementConstant(0)
MapElementConstant.one = MapElementConstant(1)


class MapElementFromFunction(MapElement):

    def __init__(self, name: str, function: Callable[[List[ExtElement]], ExtElement]):
        """
        A map defined by a callable python function.
        The number of parameters to this function is the number of standard variables for this MapElement,
        and with the same order.
        """
        self.name = name
        self.function = function
        self.num_parameters = len(inspect.signature(function).parameters)
        variables = [Var(f'X_{name}_{i}') for i in range(self.num_parameters)]
        # TODO: Maybe use the names of the variables of the original function
        super().__init__(variables)

    def to_string(self, entries: List[str]):
        entries_str = ','.join(entries)
        return f'{self.name}({entries_str})'

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        eval_entries = get_var_values(self.vars, var_dict)

        return self if eval_entries is None else CompositionFunction(function=self, entries=eval_entries)

    def _simplify_with_var_values(self, var_dict: VarDict) -> 'MapElement':

        entries = self._entry_list(var_dict)
        if all(isinstance(entry, MapElementConstant) for entry in entries):
            result = self.function(*[entry.elem for entry in entries])
            return MapElementConstant(result)

        return self._simplify_partial_constant(entries)

    # override this method instead of '_simplify_with_entries' if needed, where you can assume that
    # not all entries are constant.
    def _simplify_partial_constant(self, entries: List['MapElement']) -> 'MapElement':
        return CompositionFunction(self, entries)
