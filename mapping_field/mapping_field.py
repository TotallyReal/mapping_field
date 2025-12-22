import collections
import copy
import functools
import inspect

from abc import abstractmethod
from typing import Any, Callable, Optional, TypeVar, Union
from uuid import uuid4

from mapping_field.field import ExtElement, FieldElement
from mapping_field.log_utils.tree_loggers import TreeLogger
from mapping_field.utils.generic_properties import Property, PropertyEngine
from mapping_field.utils.processors import ProcessFailureReason, Processor, ProcessorCollection
from mapping_field.utils.serializable import DefaultSerializable
from mapping_field.utils.weakref import GenericWeakKeyDictionary

simplify_logger = TreeLogger(__name__)


def convert_to_map(elem):

    if isinstance(elem, MapElement):
        return elem

    if isinstance(elem, (int, float)) or isinstance(elem, FieldElement):
        return MapElementConstant(elem)

    if isinstance(elem, str):
        map_element = Var.try_get(elem)
        if map_element is not None:
            return map_element

        map_element = NamedFunc.try_get(elem)
        if map_element is not None:
            return map_element

    return NotImplemented


class MapElementProcessor:
    # TODO: Later change it to event registers.
    #       This is basically another function processor, where its main application was
    #       implemented in _call_with_dict, since the assignment is a type of condition.
    #       More generally we should have:
    #       function_at(func: MapElement, cond: Condition, simplify: bool)
    @abstractmethod
    def process_function(self, func: "MapElement", simplify: bool = True) -> "MapElement":
        pass


VarDict = dict["Var", "MapElement"]
FuncDict = dict["NamedFunc", "MapElement"]


def get_var_values(var_list: list["Var"], var_dict: VarDict) -> list["MapElement"] | None:
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

ClassSimplifier = Processor["MapElement"]  # ('MapElement') -> Optional['MapElement']

def class_simplifier(method) -> Callable:
    method._is_simplifier = True
    return method



class InvalidInput(Exception): pass
class ConflictingVariables(Exception): pass
class InvalidVariableOrder(Exception): pass


KeywordValue = TypeVar("KeywordValue")


def extract_keyword(kwargs, key: str, value_type: type[KeywordValue]) -> KeywordValue:
    if key not in kwargs:
        return None
    value = kwargs[key]
    if not isinstance(value, value_type):
        raise Exception(f"The {key} flag must be a {value_type}, instead got {value}")
    del kwargs[key]
    return value


SimplifierOutput = Union[ProcessFailureReason , 'MapElement' , None]

ElemPropertyEngine = PropertyEngine['MapElement', 'SimplifierContext', Property]

OutputProperties = dict[ElemPropertyEngine[Any], Any]


class SimplifierContext:
    def __init__(self, name: str | None = None):
        # id(map_element) ->  {prop_engine: prop_value}
        self.property_table = GenericWeakKeyDictionary['MapElement', dict[ElemPropertyEngine[Any], Any]]()
        self.user_property_table = GenericWeakKeyDictionary['MapElement', dict[ElemPropertyEngine[Any], Any]]()
        self.engines: list[ElemPropertyEngine[Any]] = []

    def register_engine(self, engine: ElemPropertyEngine[Any]) -> None:
        self.engines.append(engine)

    def set_property(self, element: 'MapElement', engine: ElemPropertyEngine[Property], prop_value: Property):
        if element not in self.property_table:
            self.property_table[element] = {}
        properties = self.property_table[element]

        cur_prop = properties.get(engine, None)
        properties[engine] = prop_value if cur_prop is None else engine.combine_properties(cur_prop, prop_value)

        # if engine in engine_to_promise:
        #     element.promises.add_promise(engine_to_promise[engine])

    def get_property(self, element: 'MapElement', engine: ElemPropertyEngine[Property]) -> Property | None:
        if element not in self.property_table:
            return None
        properties = self.property_table[element]
        return properties.get(engine, None)

    def get_properties(self, element: 'MapElement') -> OutputProperties:
        return self.property_table.get(element, {}).copy()

    def set_user_property(self, element: 'MapElement', engine: ElemPropertyEngine[Property], prop_value: Property):
        if element not in self.user_property_table:
            self.user_property_table[element] = {}
        properties = self.user_property_table[element]

        cur_prop = properties.get(engine, None)
        properties[engine] = prop_value if cur_prop is None else engine.combine_properties(cur_prop, prop_value)

        self.set_property(element, engine, prop_value)

    def get_user_property(self, element: 'MapElement', engine: ElemPropertyEngine[Property]) -> Property | None:
        if element not in self.user_property_table:
            return None
        properties = self.user_property_table[element]
        return properties.get(engine, None)

    def get_user_properties(self, element: 'MapElement'):
        if element not in self.user_property_table:
            return {}
        return self.user_property_table[element]

    def copy_properties(self, from_element: 'MapElement', to_element: 'MapElement'):
        from_key = id(from_element)
        if from_key not in self.property_table:
            return

        to_key = id(to_element)
        if to_key not in self.property_table:
            self.property_table[to_key] = {}

        for engine, from_prop_value in self.property_table[from_key].items():
            to_prop_value = self.property_table[to_key].get(engine, None)
            if to_prop_value is None:
                self.property_table[to_key][engine] = from_prop_value
            else:
                self.property_table[to_key][engine] = engine.combine_properties(from_prop_value, to_prop_value)

    def clear(self):
        self.property_table : dict[int, dict[ElemPropertyEngine[Any], Any]] = {}
        self.user_property_table : dict[int, dict[ElemPropertyEngine[Any], Any]] = {}


# TODO: get rid of all of these global variables
simplifier_context = SimplifierContext()


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

    _class_simplifiers = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        processor = getattr(cls, "_simplify_with_var_values")
        if processor is not None:
            processor.__name__ = f"{cls.__name__}_simplify_with_var_values"
            MapElement._simplifier.register_class_processor(cls, processor)

        # TODO: This collection process is not good, since it assumes all the simplifiers of a given class
        #       already exists at creation time. This means that if we have the following order:
        #  ;
        #           1. Define class A
        #           2. Define class B(A)
        #           3. Add simplifier to A
        #  ;
        #       Now B will not see the simplifier attached to A in (3).

        # Start with all parent rules
        cls._class_simplifiers = collections.OrderedDict()

        for base in cls.__bases__:
            if issubclass(base, MapElement):
                # copy so subclasses don’t mutate parent dict
                cls._class_simplifiers.update(base._class_simplifiers)

        # Add rules defined in this class (overriding parent rules)
        for name, obj in cls.__dict__.items():
            if getattr(obj, "_is_simplifier", False):
                cls._class_simplifiers[name] = obj

        # print(f'\nIn class {cls.__name__}')
        for simplifier in cls._class_simplifiers.values():
            # print(f'New Registering simplifier {simplifier.__qualname__}')
            cls.register_class_simplifier(simplifier)

    def __init__(self, variables: list['Var'], name: str | None = None,
                 simplified: bool = False, output_properties: OutputProperties | None = None):
        """
        The 'variables' are the ordered list used when calling the function, as in f(a_1,...,a_n).
        """
        self.name = name or self.__class__.__name__
        self._simplified_version = None if not simplified else self
        self._reset(variables, output_properties)



    def _reset(self, variables: list["Var"], output_properties: OutputProperties | None = None):
        """
        Should be called when copying this function, and want to reset it (in case you cannot go through the
        standard __init__ function).
        """
        self._hash = uuid4().int
        var_names = set(v.name for v in variables)
        if len(variables) > len(var_names):
            raise ConflictingVariables(f"Variables of functions must have distinct name, instead got: {variables}")
        self.vars = variables
        self.num_vars = len(variables)
        self._simplified_version = None
        self._simplifier.reset_element(self)

        if output_properties is not None:
            for engine, value in output_properties.items():
                # TODO: This can be called before the object is fully initialized.
                simplifier_context.set_user_property(self, engine, value)


    def __hash__(self) -> int:
        return self._hash

    # def copy(self) -> "MapElement":
    #     copied_version = copy.copy(self)
    #     copied_version.promises = self.promises.copy()
    #     return copied_version

    def set_var_order(self, variables: list["Var"]):
        """
        Reset the order of the standard variables
        """
        # TODO: update the entries in CompositionFunction
        if len(variables) > len(set(variables)):
            raise ConflictingVariables(f"Variables of functions must have distinct name, instead got: {variables}")

        if collections.Counter(variables) != collections.Counter(self.vars):
            raise InvalidVariableOrder(f"New variables order {variables} have to be on the function's variables {self.vars}")

        self.vars = variables

    # <editor-fold desc=" ------------------------ String representation ------------------------">

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.to_string({v: v.name for v in self.vars})

    def to_string(self, vars_to_str: dict["Var", str]):
        """
        --------------- Override ---------------
        Represents the function, given the string representations of its variables
        """
        if len(self.vars) == 0:
            return self.name
        vars_str = ",".join([vars_to_str[v] for v in self.vars])
        return f"{self.name}({vars_str})"

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Call function ------------------------">

    def _extract_var_dicts(self, args, kwargs) -> tuple[VarDict, FuncDict]:
        if len(args) != 0 and len(kwargs) != 0:
            raise Exception(f"When calling a function use just args or just kwargs, not both.")

        var_dict = {}
        func_dict = {}
        if len(kwargs) == 0:
            if len(args) == 1 and isinstance(args[0], dict):
                kwargs = args[0]
            else:
                if len(args) != self.num_vars:
                    raise Exception(f"Function needs to get {self.num_vars} values, and instead got {len(args)}.")
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

            raise Exception(f"Cannot assign new value to element which is not a variable of a named function : {key}")

        return var_dict, func_dict

    def __call__(self, *args, **kwargs) -> "MapElement":
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
        simplify = extract_keyword(kwargs, "simplify", bool)
        if simplify is None:
            simplify = True
        # validate_promises = extract_keyword(kwargs, "validate_promises", bool)
        # if validate_promises is None:
        #     validate_promises = False

        # TODO: Should I keep this here? In any case, handle the 'condition' parameter better.
        condition = extract_keyword(kwargs, "condition", MapElement)
        if condition is not None:
            if isinstance(condition, MapElementProcessor):
                result = condition.process_function(self, simplify=simplify)
            else:
                result = self
        else:
            var_dict, func_dict = self._extract_var_dicts(args, kwargs)
            # if validate_promises:
            #     validate_promises_var_dict(var_dict)
            result = self

            if len(func_dict) > 0 or any([v in var_dict for v in self.vars]):
                result = self._call_with_dict(var_dict, func_dict)

        return result.simplify() if simplify else result

    # Override when needed
    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> "MapElement":
        """
        Apply the map with the given values of the standard and function variables
        """
        if len(self.vars) == 0:
            return self
        entries = get_var_values(self.vars, var_dict)
        if entries is None:
            return self
        raise NotImplementedError('Wait until I rewrite the CompositionFunction class')
        # return CompositionFunction(self, entries)

    def evaluate(self) -> ExtElement | None:
        """
        Returns the constant this map defines. If it is not constant, return None.
        """
        # Avoid calling self.simplify() here, as it can cause a loop.
        return None

    def is_zero(self) -> bool:
        return self.evaluate() == 0

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Simplify 2 ------------------------">

    def simplify(self) -> "MapElement":
        return self._simplify() or self

    _simplifier = ProcessorCollection["MapElement"]()

    def _simplify(self) -> Optional["MapElement"]:
        if self._simplified_version is not None:
            return None if (self._simplified_version is self) else self._simplified_version

        path = []
        for element in MapElement._simplifier.full_process(self):
            path.append(element)
            if element._simplified_version is not None:
                if element._simplified_version is not element:
                    path.append(element._simplified_version)
                break

        if len(path) == 1:
            # The original element was already simplified
            self._simplified_version = self
            return None

        for element in path:
            element._simplified_version = path[-1]

        return path[-1]

    def is_simplified(self) -> bool:
        return self._simplified_version is self

    # Override when needed
    def _simplify_with_var_values(self) -> SimplifierOutput:
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
        return ProcessFailureReason("Trivial empty implementation", trivial=True)

    @classmethod
    def register_class_simplifier(cls, simplifier: ClassSimplifier) -> ClassSimplifier:
        MapElement._simplifier.register_class_processor(cls, simplifier)
        return simplifier

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Arithmetic functions ------------------------">

    """
    For ease of reading, all the arithmetic code is in arithmetics.py file. In particular the static functions:
        addition, subtraction, multiplication, division and negation
    are all overridden there, and _simplify_caller_function2 is updated there. 
    They are defined here to help the compiler.
    
    Each arithmetic function has 3 versions:
        - static method: (e.g. addition) Generates the actual map element for the given parameters.
        - dunder methods: (e.g. __add__) To help the coding process, and convert objects into MapElements.
        - "standard" methods: (e.g. add) Compute the function for special type of variables. These are called via
          the _simplify_caller_function2.
    In both the static and standard methods have MapElements as parameters, while the dunder methods can receive
    other types as well.    
    
    The static method generate the generic function with general simplifications (e.g. x+0 -> x).
    Override the dunder methods only to add new types of object which can be converted into MapElements.
    Override the standard methods if the arithmetic function has special behaviour for your class. For example,
    if you class represents cos^2(x) you can check if the second summand is sin^2(x) to conclude that the sum is 1.
    In case there is no special simplification, return None (or better, return the super() version of that function).
    These methods can be also "overridden" direction in _simplify_caller_function2, e.g.:
    
        if function is MapElement.addition:
          return ...
    """

    # <editor-fold desc=" ------------------------ Negation ------------------------">

    @staticmethod
    def negation(elem: "MapElement") -> "MapElement":
        return NotImplemented

    def __neg__(self) -> "MapElement":
        return MapElement.negation(self)

    def neg(self) -> Optional["MapElement"]:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Addition ------------------------">

    @staticmethod
    def addition(elem1: "MapElement", elem2: "MapElement") -> "MapElement":
        return NotImplemented

    @params_to_maps
    def __add__(self, other) -> "MapElement":
        # Very quick simplifiers:
        if other == 0:
            return self
        if self == 0:
            return other
        return MapElement.addition(self, other)

    @params_to_maps
    def __radd__(self, other) -> "MapElement":
        # Very quick simplifiers:
        if other == 0:
            return self
        if self == 0:
            return other
        return MapElement.addition(other, self)

    def add(self, other: "MapElement") -> Optional["MapElement"]:
        return None

    def radd(self, other: "MapElement") -> Optional["MapElement"]:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Subtraction ------------------------">

    @staticmethod
    def subtraction(elem1: "MapElement", elem2: "MapElement") -> "MapElement":
        return NotImplemented

    @params_to_maps
    def __sub__(self, other) -> "MapElement":
        # Very quick simplifiers:
        if other == 0:
            return self
        if self == 0:
            return MapElement.negation(other)
        return MapElement.subtraction(self, other)

    @params_to_maps
    def __rsub__(self, other) -> "MapElement":
        # Very quick simplifiers:
        if self == 0:
            return other
        if other == 0:
            return MapElement.negation(self)
        return MapElement.subtraction(other, self)

    def sub(self, other: "MapElement") -> Optional["MapElement"]:
        # Very quick simplifiers:
        if other == 0:
            return self
        return None

    def rsub(self, other: "MapElement") -> Optional["MapElement"]:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Multiplication ------------------------">

    @staticmethod
    def multiplication(elem1: "MapElement", elem2: "MapElement") -> "MapElement":
        """
        A default implementation, overridden in arithmetics.py
        """
        return NotImplemented

    @params_to_maps
    def __mul__(self, other) -> "MapElement":
        # Very quick simplifiers:
        value = other.evaluate()
        if value == 1:
            return self
        if value == 0:
            return MapElementConstant.zero
        if value == -1:
            return MapElementConstant.negation(self)

        value = self.evaluate()
        if value == 1:
            return other
        if value == 0:
            return MapElementConstant.zero
        if value == -1:
            return MapElementConstant.negation(other)

        return MapElement.multiplication(self, other)

    @params_to_maps
    def __rmul__(self, other) -> "MapElement":
        return self.__mul__(other)

    def mul(self, other: "MapElement") -> Optional["MapElement"]:
        return None

    def rmul(self, other: "MapElement") -> Optional["MapElement"]:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Division ------------------------">

    @staticmethod
    def division(elem1: "MapElement", elem2: "MapElement") -> "MapElement":
        """
        A default implementation, overridden in arithmetics.py
        """
        return NotImplemented

    @params_to_maps
    def __truediv__(self, other) -> "MapElement":
        return MapElement.division(self, other)

    @params_to_maps
    def __rtruediv__(self, other) -> "MapElement":
        return MapElement.division(other, self)

    def div(self, other: "MapElement") -> Optional["MapElement"]:
        return None

    def rdiv(self, other: "MapElement") -> Optional["MapElement"]:
        return None

    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Binary ------------------------">

    """
    These are like the arithmetic functions, but for condition (=binary) functions. Similarly, they are 
    implemented in conditions.py file. 
    """

    # <editor-fold desc=" ------------------------ Inversion ------------------------">

    @staticmethod
    def inversion(condition: "MapElement") -> "MapElement":
        raise NotImplementedError()

    def __invert__(self) -> "MapElement":
        return MapElement.inversion(self)

    def invert(self) -> Optional["MapElement"]:
        return None

    # </editor-fold>

    # <editor-fold desc=" ------------------------ And ------------------------">

    @staticmethod
    def intersection(condition1: "MapElement", condition2: "MapElement") -> "MapElement":
        raise NotImplementedError()

    def __and__(self, condition: "MapElement") -> "MapElement":
        return MapElement.intersection(self, condition)

    def and_(self, condition: "MapElement") -> SimplifierOutput:
        return ProcessFailureReason("and_ is not implemented", trivial=True)

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Or ------------------------">

    @staticmethod
    def union(condition1: "MapElement", condition2: "MapElement") -> "MapElement":
        raise NotImplementedError()

    def __or__(self, condition: "MapElement") -> "MapElement":
        return MapElement.union(self, condition)

    def or_(self, condition: "MapElement") -> SimplifierOutput:
        return ProcessFailureReason("or_ is not implemented", trivial=True)

    # </editor-fold>

    # </editor-fold>

    # <editor-fold desc=" ------------------------ Comparison condition ------------------------">

    """
    All the comparison methods are implemented in the ranged_condition.py file. They return a RangedCondition,
    and not a bool. The left shift operator `elem << n` returns the condition element `elem == n` (in contrast to
    __eq__ which returns a bool which compares the two functions). 
    For now, these are only defined for comparison with an integer.
    """

    def __le__(self, n: float) -> "MapElement":
        raise NotImplementedError()

    def __lt__(self, n: float) -> "MapElement":
        raise NotImplementedError()

    def __ge__(self, n: float) -> "MapElement":
        raise NotImplementedError()

    def __gt__(self, n: float) -> "MapElement":
        raise NotImplementedError()

    def __lshift__(self, n: float) -> "MapElement":
        raise NotImplementedError()

    # </editor-fold>


class CompositeElement(MapElement):
    """
    MapElement where the computation is done via a computation tree.
    The class more or less indicates the root, and the first level is saved in the
    'operands' variable.
    """

    def __init__(
            self, operands: list[MapElement], name: str | None = None, simplified: bool = False,
            output_properties: OutputProperties | None = None) -> None:

        self.operands = operands
        super().__init__(variables=Var.extract_variables(operands), name=name, simplified=simplified, output_properties=output_properties)

    def copy_with_operands(self, operands: list[MapElement]) -> MapElement:
        copy_version = copy.copy(self)
        # TODO: Note that this also copies the output promises in a shallow copy
        copy_version.operands = operands
        copy_version._reset(
            Var.extract_variables(operands), output_properties=simplifier_context.get_user_properties(self))
        return copy_version

    def to_string(self, vars_to_str: dict["Var", str]):
        """
        --------------- Override ---------------
        Represents the function, given the string representations of its variables
        """
        if len(self.operands) == 0:
            return self.name
        entries_str = ",".join([entry.to_string(vars_to_str) for entry in self.operands])
        return f"{self.name}({entries_str})"

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        """
        Apply the map with the given values of the standard and function variables
        """
        if len(func_dict) == 0 and set(var_dict.keys()).isdisjoint(set(self.vars)):
            return self

        new_operands = [entry._call_with_dict(var_dict, func_dict) for entry in self.operands]
        if all(new_op is op for new_op,op in zip(new_operands, self.operands)):
            return self

        return self.copy_with_operands(operands=new_operands)

    @class_simplifier
    @staticmethod
    def _entries_simplifier(elem: 'CompositeElement') -> SimplifierOutput:
        assert isinstance(elem, CompositeElement)
        simplified_entries = [entry._simplify() for entry in elem.operands]
        if all(entry is None for entry in simplified_entries):
            return ProcessFailureReason('All the entries are already simplified', trivial=True)
        simplified_entries = [simp_entry or entry for simp_entry, entry in zip(simplified_entries, elem.operands)]
        return elem.copy_with_operands(operands=simplified_entries)



# <editor-fold desc=" ---------------------- Standard and Function Variables ---------------------- ">


class Var(MapElement, DefaultSerializable):
    """
    A single variable. Can be thought of as the projection map on a variable, namely (x_1,...,x_i,...,x_n) -> x_i.
    The variable projected on is given by the name in the constructor.

    Cannot generate two variables with the same name. Trying to do so, will return the same variable.
    """

    _instances: dict[str, "Var"] = {}

    @classmethod
    def try_get(cls, var_name: str) -> Optional["Var"]:
        """
        Checks if there is a variable with the given name. Return it if exists, and otherwise None.
        """
        return cls._instances.get(var_name, None)

    # TODO: Consider using __class_getitem__ for the try_get method

    @classmethod
    def try_get_valid_assignment(cls, key, value) -> tuple["Var", MapElement] | None:
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

    @staticmethod
    def extract_variables(elements: list[MapElement]) -> list['Var']:
        seen = set()
        variables = []

        for element in elements:
            variables += [v for v in element.vars if v not in seen]
            seen.update(element.vars)
        return variables

    def __init__(self, name: str, output_properties: OutputProperties | None = None):
        """
        Initializes the Variable. If a Variable with the given name already exists, will not create a
        second object, and instead returns the existing variable.
        """
        if hasattr(self, "initialized"):
            return
        super().__init__([self], name, simplified=True, output_properties=output_properties)
        self.initialized = True

    def to_string(self, vars_to_str: dict["Var", str]):
        entries = [vars_to_str.get(v, v) for v in self.vars]
        return entries[0]

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:
        value = var_dict.get(self, None)
        if value is None:
            return self

        for engine, prop_value in simplifier_context.get_properties(self).items():
            assigned_prop = engine.compute(value, simplifier_context)
            if (assigned_prop is None) or (not engine.is_stronger_property(assigned_prop, prop_value)):
                raise InvalidInput(f"{self}={value} does not satisfy the promise of {engine}={prop_value}")
        return value

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    __hash__ = MapElement.__hash__

    def _simplify_with_var_values(self) -> SimplifierOutput:
        return None


class NamedFunc(CompositeElement, DefaultSerializable):
    """
    A named function, which can be assigned later to another function.

    Cannot generate two functions with the same name. Trying to do so, will raise an exception.
    """

    _instances: dict[str, "NamedFunc"] = {}

    @classmethod
    def try_get(cls, func_name: str) -> Optional["NamedFunc"]:
        return cls._instances.get(func_name, None)

    @classmethod
    def try_get_valid_assignment(cls, key, value) -> tuple["NamedFunc", MapElement] | None:
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
                f"Cannot assign function {func} with {func.num_vars} variables to "
                f"{value} with {value.num_vars} variables."
            )

        return func, value

    @classmethod
    def clear_vars(cls):
        cls._instances = {}

    def __new__(cls, func_name: str, variables: list[Var]):
        if func_name in cls._instances:
            cur_instance = cls._instances[func_name]
            if cur_instance.vars != variables:
                # TODO: Consider creating a specified exception
                raise Exception(f"Cannot create two functions with the same name {func_name}")
            return cur_instance

        instance = super(NamedFunc, cls).__new__(cls)
        cls._instances[func_name] = instance
        return instance

    def __init__(self, func_name: str, variables: list[Var]):
        if hasattr(self, "initialized"):
            return
        super().__init__(operands=variables, name=func_name, simplified=True)
        self.initialized = True

    @classmethod
    def serialization_name_conversion(cls) -> dict:
        return {"func_name": "name", "variables": "vars"}

    def _call_with_dict(self, var_dict: VarDict, func_dict: FuncDict) -> MapElement:

        func = func_dict.get(self, self)
        if func != self:
            return func._call_with_dict(var_dict, {})

        return super()._call_with_dict(var_dict, func_dict)


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
            raise Exception(f"The name {self.name} was already assigned to a function")

        # transform variables to Var
        actual_vars = []
        for v in variables:
            if isinstance(v, Var):
                actual_vars.append(v)
                continue
            if isinstance(v, str):
                actual_vars.append(Var(v))
                continue
            raise Exception(f"Could not define the function {self.name}: Variable {v} is not well defined.")

        # TODO: why composition
        self.assigned = NamedFunc(self.name, actual_vars)
        return self.assigned


# </editor-fold>


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
        super().__init__([], str(elem), simplified=True)
        self.elem = elem

    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, FieldElement):
            return self.elem == other
        if isinstance(other, MapElementConstant):
            return self.elem == other.elem
        return super().__eq__(other)

    __hash__ = MapElement.__hash__

    def __call__(self, *args, **kwargs):
        return self

    def evaluate(self) -> ExtElement:
        return self.elem


MapElementConstant.zero = MapElementConstant(0)
MapElementConstant.one = MapElementConstant(1)


class CompositeElementFromFunction(CompositeElement):

    def __init__(
            self, name: str, function: Callable[[list[ExtElement]], ExtElement],
            operands: list[MapElement] | None = None, simplified: bool = False,
            output_properties: OutputProperties | None = None):
        """
        A map defined by a callable python function.
        The number of parameters to this function is the number of standard variables for this MapElement,
        and with the same order.
        """
        self.function = function
        self.num_parameters = len(inspect.signature(function).parameters)
        assert self.num_parameters > 0
        if operands is None:
            operands = [Var(f"X_{name}_{i}") for i in range(self.num_parameters)]
        else:
            assert len(operands) == self.num_parameters
        super().__init__(operands=operands, name=name, simplified=simplified, output_properties = output_properties)

    # Override when needed
    def _simplify_with_var_values(self) -> MapElement | None:

        values = [operand.evaluate() for operand in self.operands]
        if any(value is None for value in values):
            return None

        result = self.function(*values)
        return MapElementConstant(result)
