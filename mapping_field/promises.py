from mapping_field.log_utils.tree_loggers import TreeLogger, green
from mapping_field.mapping_field import (
    CompositeElement, MapElement, OutputValidator, SimplifierOutput, Var,
)
from mapping_field.utils.processors import ProcessFailureReason
from mapping_field.utils.serializable import DefaultSerializable

simplify_logger = TreeLogger(__name__)


def register_promise_preserving_functions(promise: OutputValidator, elem_classes: tuple[type[CompositeElement]]):

    def _promise_preserving_simplifier(elem: MapElement) -> SimplifierOutput:
        """
            f(promise, promise) => promise
        """
        assert isinstance(elem, CompositeElement)
        assert isinstance(elem, elem_classes)

        if elem.promises.has_promise(promise) is not None:
            return ProcessFailureReason(f"{promise} is already known for {elem}", trivial=True)

        if all(operand.has_promise(promise) for operand in elem.operands):
            elem.promises.add_promise(promise)
            simplify_logger.log(f'Adding {green(promise)} promise to {green(elem)}')
            return elem

        return None

    for elem_class in elem_classes:
        assert issubclass(elem_class, CompositeElement)
        elem_class.register_class_simplifier(_promise_preserving_simplifier)


# <editor-fold desc=" ----- Condition ------">

IsCondition = OutputValidator("Condition")


# def validate_constant_condition(elem: MapElement) -> bool | None:
#     value = elem.evaluate()
#     if value is None:
#         return None
#     return value in (0, 1)
#
#
# IsCondition.register_validator(validate_constant_condition)

# </editor-fold>



# <editor-fold desc=" ----- Integral ------">


IsIntegral = OutputValidator("Integral")


@IsIntegral.register_validator
def validate_constant_integral(elem: MapElement) -> bool | None:
    value = elem.evaluate()
    if value is None:
        return None
    return int(value) == value


# @IsIntegral.register_validator
# def condition_is_integral(elem: MapElement) -> bool | None:
#     return True if elem.has_promise(IsCondition) else None


class IntVar(Var, DefaultSerializable):

    def __new__(cls, var_name: str, *args, **kwargs):
        return super(IntVar, cls).__new__(cls, var_name)

    def __init__(self, name: str):
        super().__init__(name)
        self.promises.add_promise(IsIntegral)

# </editor-fold>
