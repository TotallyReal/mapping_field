

from mapping_field.mapping_field import (
    MapElement, PropertyEngine, SimplifierContext, simplifier_context, OutputValidator,
)
from mapping_field.promises import IsIntegral, IsCondition

class BoolPropertyEngine(PropertyEngine[bool]):

    def __init__(self, validator: OutputValidator):
        self.validator = validator

    def compute(self, element: MapElement, context: SimplifierContext) -> bool | None:
        value = simplifier_context.get_property(element, self)
        if value is not None:
            return value
        if element.has_promise(self.validator):
            return True

        return None

    def combine_properties(self, prop1: bool, prop2: bool) -> bool:
        assert prop1 == prop2
        return prop1

    def is_stronger_property(self, strong_prop: bool, weak_prop: bool) -> bool:
        return strong_prop == weak_prop # TODO: This is not exactly true, but keep as is for now

is_condition = BoolPropertyEngine(IsCondition)

is_integral = BoolPropertyEngine(IsIntegral)