from abc import abstractmethod
import collections
import inspect
from typing import Callable, Dict, List, Optional
from field import FieldElement, ExtElement


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

    def set_var_order(self, variables: List['Var']):
        if len(variables) > len(set(variables)):
            raise Exception(f'Function must have distinct variables')

        if collections.Counter(variables) != collections.Counter(self.vars):
            raise Exception(f'New variables order {variables} have to be on the function\'s variables {self.vars}')

        self.vars = variables

    def __call__(self, *args, **kwargs) -> 'MapElement':
        simplify = True
        if 'simplify' in kwargs:
            simplify = kwargs['simplify']
            if not isinstance(simplify, bool):
                raise Exception(f'The "simplify" flag must be a boolean, instead got {simplify}')
            del kwargs['simplify']

        var_dict = {}
        func_dict = {}
        if len(kwargs) == 0:
            if len(args) != self.num_vars:
                raise Exception(f'Function needs to get {self.num_vars} values, and instead got {len(args)}.')
            var_dict = {v: convert_to_map(value) for v, value in zip(self.vars, args)}
            args = []

        if len(args) != 0:
            raise Exception(f'When calling a function use just args or just kwargs, not both.')

        # Split assignments into variables and functions
        for key, value in kwargs.items():
            v = Var.try_get(key)
            if v is not None:
                var_dict[v] = convert_to_map(value)
                continue

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
        return self

    def evaluate(self) -> ExtElement:
        map_elem = self.simplify()
        assert isinstance(map_elem, MapElementConstant)
        return map_elem.evaluate()

    def simplify(self) -> 'MapElement':
        """
        Try to simplify the given function (e.g. 1+x*0+y -> 1+y)
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
        # return 'empty'
        vars_str_list = [var.name for var in self.vars]
        return self.to_string(vars_str_list)

    @abstractmethod
    def to_string(self, vars_str_list: List[str]):
        """
        --------------- Override ---------------
        Represents the function, given the string representations of its variables
        """
        pass

    # Overriding the following functions in the arithmetics.py file.
    # Adding them here to help the compiler know that they exist.

    def __add__(self, other):
        return NotImplemented

    def __radd__(self, other):
        return NotImplemented

    def __neg__(self):
        return NotImplemented

    def __sub__(self, other):
        return NotImplemented

    def __rsub__(self, other):
        return NotImplemented

    def __mul__(self, other):
        return NotImplemented

    def __rmul__(self, other):
        return NotImplemented

    def __truediv__(self, other):
        return NotImplemented

    def __rtruediv__(self, other):
        return NotImplemented


class Var(MapElement):
    """
    A single variable. Can be thought of as the projection map on a variable, namely (x_1,...,x_i,...,x_n) -> x_n.
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

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        # Try to look both for the variable itself, and its name
        return var_dict.get(self, self)


class NamedFunc(MapElement):
    """
    A named function, which can be assigned later to another function.

    Cannot generate two functions with the same name. Trying to do so, will raise an exception.
    """
    _instances: Dict[str, 'NamedFunc'] = {}

    @classmethod
    def try_get(cls, func_name: str) -> Optional['NamedFunc']:
        return cls._instances.get(func_name, None)

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

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        if len(func_dict) == 0:
            eval_entries = []
            compose = False
            for var in self.vars:
                eval_var = var_dict.get(var, var)
                eval_entries.append(eval_var)
                if eval_var != var:
                    compose = True
            return CompositionFunction(function=self, entries=eval_entries) if compose else self

        return func_dict.get(self, self)._call_with_dict(var_dict, {})

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        if all(var == entry for var, entry in zip(self.vars, simplified_entries)):
            return self
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

        # TODO: why composition
        self.assigned = NamedFunc(self.name, actual_vars)
        return self.assigned


class CompositionFunction(MapElement):

    def __init__(self, function: MapElement, entries: List[MapElement]):
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

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        if len(var_dict) == 0 and len(func_dict) == 0:
            return self
        eval_function = self.function._call_with_dict({}, func_dict)
        eval_entries = [entry._call_with_dict(var_dict, func_dict) for entry in self.entries]

        return CompositionFunction(function=eval_function, entries=eval_entries)

    def to_string(self, vars_str_list: List[str]):
        # Compute the str representation for each entry, by supplying it the str
        # representations of its variables
        var_str_dict = {var: var_str for var, var_str in zip(self.vars, vars_str_list)}
        entries_str_list = [
            entry.to_string([var_str_dict[var] for var in entry.vars])
            for entry in self.entries]
        return self.function.to_string(entries_str_list)

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        # Compute the simplified entries, by supplying each with the simplified version
        # of its variables
        simplified_var_dict = {var: simplified_var for var, simplified_var in zip(self.vars, simplified_entries)}
        simplified_entries = [
            entry._simplify_with_entries([simplified_var_dict[var] for var in entry.vars])
            for entry in self.entries]
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

    def to_string(self, vars_str_list: List[str]):
        return str(self.elem)

    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, FieldElement):
            return self.elem == other
        if isinstance(other, MapElementConstant):
            return self.elem == other.elem
        return super().__eq__(other)


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

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        if len(var_dict) == 0:
            return self
        eval_entries = []
        compose = False
        for var in self.vars:
            eval_var = var_dict.get(var, var)
            eval_entries.append(eval_var)
            if eval_var != var:
                compose = True

        if not compose:
            return self

        return CompositionFunction(function=self, entries=eval_entries)

    def _simplify_with_entries(self, simplified_entries: List['MapElement']) -> 'MapElement':
        if any(not isinstance(entry, MapElementConstant) for entry in simplified_entries):
            return self._simplify_partial_constant(simplified_entries)

        result = self.function(*[entry.elem for entry in simplified_entries])
        return MapElementConstant(result)

    def _simplify_partial_constant(self, simplified_entries: List['MapElement']) -> 'MapElement':
        return CompositionFunction(self, simplified_entries)
