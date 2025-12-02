import inspect
import types
from abc import ABC
from typing import Callable, Type

from mapping_field.mapping_field import (
    MapElement, PropertyEngine, SimplifierContext, Property,
    CompositeElement,
)
from mapping_field.utils.generic_properties import PropertyByRulesEngine, property_rule


class BoolPropertyEngine(PropertyByRulesEngine[MapElement, SimplifierContext, bool]):

    def __init__(self):
        super().__init__()

        self.auto_classes: list[type[MapElement]] = []
        self.property_preserving_classes: list[type[CompositeElement]] = []

    def compute(self, element: MapElement, context: SimplifierContext) -> bool | None:
        value = context.get_property(element, self)
        if value is not None:
            return value

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

    def add_auto_class(self, cls: Type[MapElement] | list[Type[MapElement]]):
        if not isinstance(cls, list):
            cls = [cls]
        self.auto_classes.extend(cls)

    @property_rule
    def check_auto_classes(self, element: MapElement, context: SimplifierContext) -> bool | None:
        for cls in self.auto_classes:
            if isinstance(element, cls):
                return True
        return None

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


# <editor-fold desc="Condition">

class ConditionEngine(BoolPropertyEngine):

    @staticmethod
    @property_rule
    def zero_one_is_condition(element: MapElement, context: SimplifierContext) -> bool | None:
        value = element.evaluate()
        if value is None:
            return None

        if isinstance(value, int):
            return value in (0, 1)

        if isinstance(value, float):
            return value == int(value) and int(value) in (0, 1)

        return None

is_condition = ConditionEngine()

# </editor-fold>


# <editor-fold desc="Integral">

class IntegralEngine(BoolPropertyEngine):

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

# </editor-fold>