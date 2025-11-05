from mapping_field.mapping_field import MapElement
from mapping_field.new_code.validators import MultiValidator

IsCondition = MultiValidator[MapElement]("Condition")
IsIntegral = MultiValidator[MapElement]("Integral")

def validate_constant_integral(elem: MapElement) -> bool:
    value = elem.evaluate()
    return (value is not None) and (int(value) == value)

IsIntegral.register_validator(validate_constant_integral)