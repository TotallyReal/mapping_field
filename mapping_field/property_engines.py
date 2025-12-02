import inspect
import types
from abc import ABC
from typing import Callable, Type

from mapping_field.mapping_field import (
    MapElement, PropertyEngine, SimplifierContext, simplifier_context, OutputValidator, engine_to_promise, Property,
    CompositeElement,
)
from mapping_field.promises import IsIntegral, IsCondition

PropertyRule = Callable[[MapElement, SimplifierContext], Property | None]

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

class PropertyByRulesEngine(PropertyEngine[Property], ABC):

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

    def register_rule(self, method: PropertyRule[bool]):
        self._rules.append(method)


class BoolPropertyEngine(PropertyByRulesEngine[bool]):

    def __init__(self, validator: OutputValidator):
        super().__init__()
        self.validator = validator

        self.property_preserving_classes: list[type[CompositeElement]] = []

    def __str__(self):
        return str(self.validator)

    def compute(self, element: MapElement, context: SimplifierContext) -> bool | None:
        value = context.get_property(element, self)
        if value is not None:
            return value
        if element.has_promise(self.validator):
            return True

        for rule in self._rules:
            result = rule(element, context)
            if result is not None:
                context.set_property(element, self, result)
                return result

        return None

    def combine_properties(self, prop1: bool, prop2: bool) -> bool:
        assert prop1 == prop2
        return prop1

    def is_stronger_property(self, strong_prop: bool, weak_prop: bool) -> bool:
        return strong_prop == weak_prop # TODO: This is not exactly true, but keep as is for now

    def add_property_preserving_class(self, cls: Type[MapElement] | list[Type[MapElement]]):
        if not isinstance(cls, list):
            cls = [cls]
        self.property_preserving_classes.extend(cls)

    @property_rule
    def property_preserving(self, element: MapElement, context: SimplifierContext) -> bool | None:
        if not isinstance(element, tuple(self.property_preserving_classes)):
            return None

        operands_values = [self.compute(operand, context) for operand in element.operands]
        if all(operands_values):
            return True

        return None

is_condition = BoolPropertyEngine(IsCondition)
engine_to_promise[is_condition] = IsCondition

# <editor-fold desc="Integral">

class IntegralEngine(BoolPropertyEngine):

    def __init__(self):
        super().__init__(IsIntegral)

    def __getitem__(self, value: bool):
        return {self: value}

    @staticmethod
    @property_rule
    def constant_is_integral(element: MapElement, context: SimplifierContext) -> bool | None:
        value = element.evaluate()
        if value is None:
            return None

        if isinstance(value, int):
            return True

        if isinstance(value, float):
            return value == int(value)

        return None

    @staticmethod
    @property_rule
    def condition_is_integral(element: MapElement, context: SimplifierContext) -> bool | None:
        if is_condition.compute(element, context):
            return True

        return None

is_integral = IntegralEngine()
engine_to_promise[is_integral] = IsIntegral

# </editor-fold>