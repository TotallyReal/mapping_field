import collections
import functools
import inspect
from typing import Callable, Dict, List, Optional, Tuple, Iterator, Type, TypeVar

from mapping_field.processors import ProcessorCollection, Processor, ParamProcessor
from mapping_field.field import FieldElement, ExtElement
from mapping_field.serializable import DefaultSerializable
from mapping_field.tree_loggers import TreeLogger, TreeAction, red, cyan, magenta, green

simplify_logger = TreeLogger(__name__)

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
        eval_var = var_dict.get(var, None)
        eval_entries.append(eval_var or var)
        if eval_var is not None:
            trivial = False

    return None if trivial else eval_entries

def params_to_maps(f):
    @functools.wraps(f)
    def wrapper(self, element):
        value = convert_to_map(element)
        return NotImplemented if value is NotImplemented else f(self, value)

    return wrapper

# Suppose that I have some function H(x,y), and we want to compute H(x0, y0) for some specific x0, y0. There
# are 2 main ways how to simplify this expression. For example, consider the function H(x,y) = x + y, then:
#  1. General simplification, independent of the choice of x, y. For example, for x0=0, we have
#           H(0, y0) = 0 + y0 = y0.
#  2. Specific simplification for the choice of x0, y0, for example:
#         H(sin^2(x) , cos^2(x)) = sin^2(x) + cos^2(x) = 1
#
# Hence, in general the process of simplifying H(x0, y0) works as follows:
#
# 1. Simplify the entries x0 and y0.
# 2. Check if there is a general simplification.
# 3. If not, check is x0 knows how to simplify H (x0.special_simplification(H, position=0, y0)).
# 4. If not, check is y0 knows how to simplify H (y0.special_simplification(H, position=1, x0)).
# 5. If no simplification was found, save as generic (Composition) H(x0, y0).

# TODO: find how to switch the order
ElemSimplifier = ParamProcessor[VarDict, 'MapElement']
ClassSimplifier = Processor['MapElement', VarDict]

class OutputPromise:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

T = TypeVar('T', bound='MapElement')

def validate_promises_var_dict(var_dict: VarDict) -> bool:
    for v, value in var_dict.items():
        for promise in v.output_promises():
            assert value.has_promise(promise), f'{v}={value} does not satisfy the promise of {promise}'

def always_validate_promises(cls: Type[T]) -> Type[T]:

    original_call_method = cls.__call__

    def call_wrapper(self, *args, **kwargs):
        if 'validate_promises' not in kwargs:
            kwargs['validate_promises'] = True
        return original_call_method(self, *args, **kwargs)

    cls.__call__ = call_wrapper
    return cls

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


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        processor = cls.__dict__.get('_simplify_with_var_values2', None)
        if processor is not None:
            MapElement._simplifier.register_class_processor(cls, processor)

    def __init__(self, variables: List['Var'], name: Optional[str] = None):
        """
        The 'variables' are the ordered list used when calling the function, as in f(a_1,...,a_n).
        """
        self.name = name or self.__class__.__name__
        var_names = set(v.name for v in variables)
        if len(variables) > len(var_names):
            raise Exception(f'Function must have distinct variables: {variables}')
        self.vars = variables
        self.num_vars = len(variables)
        self._simplified = False
        self._promises = set()

    def set_var_order(self, variables: List['Var']):
        """
        Reset the order of the standard variables
        """
        if len(variables) > len(set(variables)):
            raise Exception(f'Function must have distinct variables')

        if collections.Counter(variables) != collections.Counter(self.vars):
            raise Exception(f'New variables order {variables} have to be on the function\'s variables {self.vars}')

        self.vars = variables

    # <editor-fold desc=" ------------------------ Output Promises ------------------------ ">

    def add_promise(self, promise: OutputPromise):
        self._promises.add(promise)

    def output_promises(self) -> Iterator[OutputPromise]:
        return iter(self._promises)

    def has_promise(self, promise: OutputPromise):
        return promise in self._promises

    # </editor-fold>

    # <editor-fold desc=" ------------------------ String represnetation ------------------------">

    def __repr__(self):
        return str(self)

    def __str__(self):
        vars_str_list = [var.name for var in self.vars]
        return self.to_string(vars_str_list)

    def to_string(self, vars_str_list: List[str]):
        """
        --------------- Override ---------------
        Represents the function, given the string representations of its variables
        """
        if len(vars_str_list) == 0:
            return self.name
        vars_str = ','.join(vars_str_list)
        return f'{self.name}({vars_str})'

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Call and Simplify function ------------------------">

    def _extract_var_dicts(self, args, kwargs) -> Tuple[VarDict, FuncDict]:
        if len(args) != 0 and len(kwargs) != 0:
            raise Exception(f'When calling a function use just args or just kwargs, not both.')

        var_dict = {}
        func_dict = {}
        if len(kwargs) == 0:
            if len(args) == 1 and isinstance(args[0], Dict):
                kwargs = args[0]
            else:
                if len(args) != self.num_vars:
                    raise Exception(f'Function needs to get {self.num_vars} values, and instead got {len(args)}.')
                var_dict = {v: convert_to_map(value) for v, value in zip(self.vars, args)}
                return var_dict, dict()

        # Split assignments into variables and functions
        for key, value in kwargs.items():

            key_value_pair = Var.try_get_valid_assignment(key, value)
            if key_value_pair is not None:
                var_dict[key_value_pair[0]] = key_value_pair[1]
                continue

            key_value_pair = NamedFunc.try_get_valid_assignment(key, value)
            if key_value_pair is not None:
                func_dict[key_value_pair[0]] = key_value_pair[1]
                continue

            raise Exception(f'Cannot assign new value to element which is not a variable of a named function : {key}')

        return var_dict, func_dict

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

        validate_promises = False
        if 'validate_promises' in kwargs:
            validate_promises = kwargs['validate_promises']
            if not isinstance(validate_promises, bool):
                raise Exception(f'The "validate_promises" flag must be a boolean, instead got {validate_promises}')
            del kwargs['validate_promises']

        var_dict, func_dict = self._extract_var_dicts(args, kwargs)
        if validate_promises:
            validate_promises_var_dict(var_dict)
        result = self
        if len(func_dict) > 0 or any([v in var_dict for v in self.vars]):
            result = self._call_with_dict(var_dict, func_dict)
        return result.simplify2() if simplify else result

    # Override when needed
    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        """
        Apply the map with the given values of the standard and function variables
        """
        if len(self.vars) == 0:
            return self
        entries = get_var_values(self.vars, var_dict)
        return CompositionFunction(self, entries)

    def evaluate(self) -> Optional[ExtElement]:
        """
        Returns the constant this map defines. If it is not constant, raises an error.
        """
        map_elem = self.simplify2()
        return map_elem.evaluate() if isinstance(map_elem, MapElementConstant) else None

    def is_zero(self) -> bool:
        return self.evaluate() == 0

    # <editor-fold desc=" ------------------------ Simplify 2 ------------------------">

    def simplify2(self) -> 'MapElement':
        return self._simplify2() or self

    _simplifier = ProcessorCollection['MapElement', VarDict]()

    def _simplify2(self, var_dict: Optional[VarDict] = None) -> Optional['MapElement']:
        if var_dict is None:
            var_dict = {}

        if self._simplified and len(var_dict) == 0:
            return None

        simplified_version = MapElement._simplifier.full_process(self, var_dict)

        if simplified_version is not None:
            simplified_version._simplified = True
            return simplified_version

        if len(var_dict) == 0:
            self._simplified = True
            return None

        return None

    # Override when needed
    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
        """
        --------------- Override when needed ---------------
        Try to simplify the given function, given assignment of variables.

        For example, the multiplication function by itself cannot be simplified:
            (x,y) -> x*y

        If, for example, we know that x=0, then we can simplify it to the zero function.
        If x=1, we can simplify it to the y function.
        If x=2, and y doesn't change, then the function remains a multiplication function 3*y, namely we
        can't do any simplification other than setting the variables. In this case we return None.
        """
        return None

    def _simplify_caller_function2(self, function:'MapElement', position: int, var_dict: VarDict) -> Optional['MapElement']:
        return None

    @classmethod
    def register_class_simplifier(cls, simplifier: ClassSimplifier):
        MapElement._simplifier.register_class_processor(cls, simplifier)

    def register_simplifier(self, simplifier: ElemSimplifier):
        MapElement._simplifier.register_elem_processor(self, simplifier)

    # </editor-fold>

    def _entry_list(self, var_dict: VarDict):
        return [var_dict.get(v, v) for v in self.vars]

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Arithmetic functions ------------------------">

    """
    For ease of reading, all the arithmetic code is in arithmetics.py file. In particular the static functions:
        addition, subtraction, multiplication, division and negation
    are all overriden there, and _simplify_caller_function2 is updated there. 
    They are defined here to help the compiler.
    
    Each arithmetic function has 3 versions:
        - static method: (e.g. addition) Generates the actual map element for the given parameters.
        - dunder methods: (e.g. __add__) To help the coding process, and convert objects into MapElements.
        - "standard" methods: (e.g. add) Compute the function for special type of variables. These are called via
          the _simplify_caller_function2.
    In both the static and standard methods have MapElements as parameters, while the dunder methods can recieve
    other types as well.    
    
    The static method generate the generic function with general simplifications (e.g. x+0 -> x).
    Override the dunder methods only to add new types of object which can be converted into MapElements.
    Override the standard methods if the arithmetic function has special behaviour for your class. For example,
    if you class represents cos^2(x) you can check if the second summand is sin^2(x) to conclude that the sum is 1.
    In case there is no special simplification, return None (or better, return the super() version of that function).
    These methods can be also "overriden" direction in _simplify_caller_function2, e.g.:
    
        if function is MapElement.addition:
          return ...
    """

    # <editor-fold desc=" ------------------------ Negation ------------------------">

    @staticmethod
    def negation(elem: 'MapElement') -> 'MapElement':
        return NotImplemented

    def __neg__(self) -> 'MapElement':
        return MapElement.negation(self)

    def neg(self) -> Optional['MapElement']:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Addition ------------------------">

    @staticmethod
    def addition(elem1: 'MapElement' ,elem2: 'MapElement') -> 'MapElement':
        return NotImplemented

    @params_to_maps
    def __add__(self, other) -> 'MapElement':
        # Very quick simplifiers:
        if other == 0:
            return self
        return MapElement.addition(self, other)

    @params_to_maps
    def __radd__(self, other) -> 'MapElement':
        # Very quick simplifiers:
        if other == 0:
            return self
        return MapElement.addition(other, self)

    def add(self, other: 'MapElement') -> Optional['MapElement']:
        return None

    def radd(self, other: 'MapElement') -> Optional['MapElement']:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Subtraction ------------------------">

    @staticmethod
    def subtraction(elem1: 'MapElement' ,elem2: 'MapElement') -> 'MapElement':
        return NotImplemented

    @params_to_maps
    def __sub__(self, other) -> 'MapElement':
        return MapElement.subtraction(self, other)

    @params_to_maps
    def __rsub__(self, other) -> 'MapElement':
        return MapElement.subtraction(other, self)

    def sub(self, other: 'MapElement') -> Optional['MapElement']:
        # Very quick simplifiers:
        if other == 0:
            return self
        return None

    def rsub(self, other: 'MapElement') -> Optional['MapElement']:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Multiplication ------------------------">

    @staticmethod
    def multiplication(elem1: 'MapElement' ,elem2: 'MapElement') -> 'MapElement':
        """
        A default implementation, overriden in arithmetics.py
        """
        return NotImplemented

    @params_to_maps
    def __mul__(self, other) -> 'MapElement':
        # Very quick simplifiers:
        if other == 1:
            return self
        if other == 0:
            return MapElementConstant.zero
        return MapElement.multiplication(self, other)

    @params_to_maps
    def __rmul__(self, other) -> 'MapElement':
        # Very quick simplifiers:
        if other == 1:
            return self
        if other == 0:
            return MapElementConstant.zero
        return MapElement.multiplication(other, self)

    def mul(self, other: 'MapElement') -> Optional['MapElement']:
        return None

    def rmul(self, other: 'MapElement') -> Optional['MapElement']:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Division ------------------------">

    @staticmethod
    def division(elem1: 'MapElement' ,elem2: 'MapElement') -> 'MapElement':
        """
        A default implementation, overriden in arithmetics.py
        """
        return NotImplemented

    @params_to_maps
    def __truediv__(self, other) -> 'MapElement':
        return MapElement.division(self, other)

    @params_to_maps
    def __rtruediv__(self, other) -> 'MapElement':
        return MapElement.division(other, self)

    def div(self, other: 'MapElement') -> Optional['MapElement']:
        return None

    def rdiv(self, other: 'MapElement') -> Optional['MapElement']:
        return None

    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Binary ------------------------">

    """
    These are like the arithmetic functions, but for condition (=binary) functions. Similarly, they are 
    implemented in new_conditions.py file. 
    """

    # <editor-fold desc=" ------------------------ Inversion ------------------------">

    @staticmethod
    def inversion(condition: 'MapElement') -> 'MapElement':
        raise NotImplementedError()

    def __invert__(self) -> 'MapElement':
        return MapElement.inversion(self)

    def invert(self) -> Optional['MapElement']:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ And ------------------------">

    @staticmethod
    def intersection(condition1: 'MapElement', condition2: 'MapElement') -> 'MapElement':
        raise NotImplementedError()

    def __and__(self, condition: 'MapElement') -> 'MapElement':
        return MapElement.intersection(self, condition)

    def and_(self, condition: 'MapElement') -> Optional['MapElement']:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Or ------------------------">

    @staticmethod
    def union(condition1: 'MapElement', condition2: 'MapElement') -> 'MapElement':
        raise NotImplementedError()

    def __or__(self, condition: 'MapElement') -> 'MapElement':
        return MapElement.union(self, condition)

    def or_(self, condition: 'MapElement') -> Optional['MapElement']:
        return None

    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Comparison condition ------------------------">

    """
    All the comparison methods are implemented in the condition.py file.
    For now, these are only defined for comparison with an integer.
    """

    def __le__(self, n: int) -> 'Condition':
        raise NotImplementedError()

    def __lt__(self, n: int) -> 'Condition':
        raise NotImplementedError()

    def __ge__(self, n: int) -> 'Condition':
        raise NotImplementedError()

    def __gt__(self, n: int) -> 'Condition':
        raise NotImplementedError()

    # </editor-fold>


class Var(MapElement, DefaultSerializable):
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
    def try_get_valid_assignment(cls, key, value) -> Optional[Tuple['Var', MapElement]]:
        value = convert_to_map(value)
        if value is NotImplemented:
            return None
        if isinstance(key, Var):
            return key, value
        if isinstance(key, str):
            key = cls.try_get(key)
            return None if (key is None) else (key, value)
        return None

    @classmethod
    def clear_vars(cls):
        cls._instances = {}

    def __new__(cls, name: str):
        if name in cls._instances:
            v = cls._instances[name]
            assert v.__class__ == cls, f'Attempted to create two variables of different classes with the same name {name}'
            return v

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
        super().__init__([self], name)
        self.initialized = True
        self._simplified = True

    def to_string(self, vars_str_list: List[str]):
        return self.name

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        return var_dict.get(self, self)

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(('Var', self.name))


class NamedFunc(MapElement, DefaultSerializable):
    """
    A named function, which can be assigned later to another function.

    Cannot generate two functions with the same name. Trying to do so, will raise an exception.
    """
    _instances: Dict[str, 'NamedFunc'] = {}

    @classmethod
    def try_get(cls, func_name: str) -> Optional['NamedFunc']:
        return cls._instances.get(func_name, None)

    @classmethod
    def try_get_valid_assignment(cls, key, value) -> Optional[Tuple['NamedFunc', MapElement]]:
        value = convert_to_map(value)
        if value is NotImplemented:
            return None

        func = None
        if isinstance(key, NamedFunc):
            func = key
        if isinstance(key, str):
            func = NamedFunc.try_get(key)
        if func is None:
            return None

        if func.num_vars != value.num_vars:
            raise Exception(
                f'Cannot assign function {func} with {func.num_vars} variables to '
                f'{value} with {value.num_vars} variables.')

        return func, value

    @classmethod
    def clear_vars(cls):
        cls._instances = {}

    def __new__(cls, func_name: str, variables: List[Var]):
        if func_name in cls._instances:
            cur_instance = cls._instances[func_name]
            if cur_instance.vars != variables:
                # TODO: Consider creating a specified exception
                raise Exception(f'Cannot create two functions with the same name {func_name}')
            return cur_instance

        instance = super(NamedFunc, cls).__new__(cls)
        cls._instances[func_name] = instance
        return instance

    def __init__(self, func_name: str, variables: List[Var]):
        if hasattr(self, 'initialized'):
            return
        super().__init__(variables, func_name)
        self.initialized = True
        self._simplified = True

    @classmethod
    def serialization_name_conversion(cls) -> Dict:
        return {
            'func_name': 'name',
            'variables': 'vars'
        }

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:

        func = func_dict.get(self, self)
        if func != self:
            return func._call_with_dict(var_dict, {})

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


class CompositionFunction(MapElement, DefaultSerializable):

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

        # Works under the assumption that the entries satisfy their promises for the given function
        for promise in self.function.output_promises():
            self.add_promise(promise)

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
        if (eval_function is self.function) and all([e1 is e2 for e1, e2 in zip(self.entries, eval_entries)]):
            return self

        return CompositionFunction(function=eval_function, entries=eval_entries)

    # TODO: When simplifying arithmetic function, e.g. a + b, after simplifying both a and b,
    #       we should check if it has a new type of arithmetic function that we can call.

    # Override when needed
    def _simplify_with_var_values2(self, var_dict: Optional[VarDict] = None) -> Optional['MapElement']:
        simplify_logger.log('Simplifying just the function')
        function: MapElement = self.function.simplify2()
        simplify_logger.log('Simplifying just the entries')
        simplified_entries = [entry._simplify2(var_dict) for entry in self.entries]

        is_simpler = (function is not self.function) | any([entry is not None for entry in simplified_entries])
        simplified_entries = [simp_entry or entry for simp_entry, entry in zip(simplified_entries, self.entries)]

        simplified_entries_dict  = {v : entry
                                    for v, entry in zip(function.vars, simplified_entries)}
        simplify_logger.log('Simplifying function with entries')
        result = function._simplify2(simplified_entries_dict)
        if result is not None:
            return result

        # TODO: consider moving it into a simplifier processor, since it is mainly used for arithmetics
        simplify_logger.log('Simplifying via positional entries')
        for position, v in enumerate(function.vars):
            pos_entry = simplified_entries_dict[v]
            simplify_logger.log(f'Pos {red(v)} -> {red(pos_entry)} with class {cyan(pos_entry.__class__.__name__)}', TreeAction.GO_DOWN)
            result = pos_entry._simplify_caller_function2(function, position, simplified_entries_dict)

            if result is not None:
                simplify_logger.log(f'Pos {green(result)}', TreeAction.GO_UP)
                return result
            simplify_logger.log(f'Pos {magenta("& & &")}', TreeAction.GO_UP)

        if is_simpler:
            return CompositionFunction(function, simplified_entries)

        return None


class MapElementConstant(MapElement, DefaultSerializable):
    """
    Used for constant maps, and for casting elements into maps.
    """

    zero = None
    one = None

    def __new__(cls, elem: ExtElement):
        if elem == 0 and MapElementConstant.zero is not None:
            return MapElementConstant.zero
        if elem == 1 and MapElementConstant.one is not None:
            return MapElementConstant.one
        return super().__new__(cls)

    def __init__(self, elem: ExtElement):
        super().__init__([], str(elem))
        self.elem = elem
        self._simplified = True

    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, FieldElement):
            return self.elem == other
        if isinstance(other, MapElementConstant):
            return self.elem == other.elem
        return super().__eq__(other)

    def __call__(self, *args, **kwargs):
        return self

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
        self.function = function
        self.num_parameters = len(inspect.signature(function).parameters)
        variables = [Var(f'X_{name}_{i}') for i in range(self.num_parameters)]
        # TODO: Maybe use the names of the variables of the original function
        super().__init__(variables, name)

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> 'MapElement':
        eval_entries = get_var_values(self.vars, var_dict)

        return self if eval_entries is None else CompositionFunction(function=self, entries=eval_entries)

    # Override when needed
    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional['MapElement']:
        entries = get_var_values(self.vars, var_dict)
        if entries is None:
            return None

        if all(isinstance(entry, MapElementConstant) for entry in entries):
            result = self.function(*[entry.elem for entry in entries])
            return MapElementConstant(result)

        return None

