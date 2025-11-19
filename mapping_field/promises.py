from typing import Optional

from mapping_field.mapping_field import MapElement, OutputValidator, Var, always_validate_promises
from mapping_field.utils.serializable import DefaultSerializable

IsCondition = OutputValidator("Condition")


def validate_constant_condition(elem: MapElement) -> Optional[bool]:
    value = elem.evaluate()
    if value is None:
        return None
    return value in (0, 1)


IsCondition.register_validator(validate_constant_condition)

IsIntegral = OutputValidator("Integral")


def validate_constant_integral(elem: MapElement) -> Optional[bool]:
    value = elem.evaluate()
    if value is None:
        return None
    return int(value) == value

def condition_is_integral(elem: MapElement) -> Optional[bool]:
    return True if elem.has_promise(IsCondition) else None


IsIntegral.register_validator(validate_constant_integral)
IsIntegral.register_validator(condition_is_integral)


@always_validate_promises
class IntVar(Var, DefaultSerializable):

    def __new__(cls, var_name: str, *args, **kwargs):
        return super(IntVar, cls).__new__(cls, var_name)

    def __init__(self, name: str):
        super().__init__(name)
        self.promises.add_promise(IsIntegral)
