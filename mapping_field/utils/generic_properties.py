import inspect
import types

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

Property = TypeVar("Property")
Element = TypeVar("Element")
Context = TypeVar("Context")


class PropertyEngine(Generic[Element, Context, Property], ABC):

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def compute(self, element: Element, context: Context) -> Property | None:
        raise NotImplementedError()

    @abstractmethod
    def combine_properties(self, prop1: Property, prop2: Property) -> Property:
        raise NotImplementedError()

    @abstractmethod
    def is_stronger_property(self, strong_prop: Property, weak_prop: Property) -> bool:
        """
        Checks whether the strong_prop is stronger than (namely, implying the) weak_prop.

        Example:
             in_range[3,6] is stronger than in_range[1,10].
        """
        raise NotImplementedError()


PropertyRule = Callable[[Element, Context], Property | None]

def property_rule(method) -> Callable:
    params = list(inspect.signature(method).parameters.values())
    assert len(params) in (2,3)

    if params[0].name == "self":
        # standardmethod:   (self, element, context)
        method._compute_type = "bound"
    else:
        # staticmethod:     (element, context)
        method._compute_type = "static"
    return method

class PropertyByRulesEngine(PropertyEngine[Element, Context, Property], ABC):

    def __init__(self):
        self._rules = []

        cls = self.__class__
        for name in dir(cls):  # walks superclasses automatically
            attr = getattr(cls, name)
            mode = getattr(attr, "_compute_type", None)
            if mode == "bound":
                self.register_rule(types.MethodType(attr, self))
            elif mode == "static":
                self.register_rule(attr)

    def register_rule(self, method: PropertyRule[Element, Context, Property]):
        self._rules.append(method)
        return method