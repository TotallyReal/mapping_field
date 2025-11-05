from mapping_field.mapping_field import MapElement
from mapping_field.new_code.mapping_field import OutputValidator, Var, always_validate_promises
from mapping_field.serializable import DefaultSerializable

IsCondition = OutputValidator("Condition")

def validate_constant_condition(elem: MapElement) -> bool:
    value = elem.evaluate()
    return value in (0,1)

IsCondition.register_validator(validate_constant_condition)


IsIntegral = OutputValidator("Integral")

def validate_constant_integral(elem: MapElement) -> bool:
    value = elem.evaluate()
    return (value is not None) and (int(value) == value)

IsIntegral.register_validator(validate_constant_integral)

@always_validate_promises
class IntVar(Var, DefaultSerializable):

    def __new__(cls, var_name: str):
        return super(IntVar, cls).__new__(cls, var_name)

    def __init__(self, name: str):
        super().__init__(name)
        self.promises.add_promise(IsIntegral)

@always_validate_promises
class BoolVar(Var, DefaultSerializable):

    def __new__(cls, var_name: str):
        return super(BoolVar, cls).__new__(cls, var_name)

    def __init__(self, name: str):
        super().__init__(name)
        self.promises.add_promise(IsCondition)