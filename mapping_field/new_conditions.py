from typing import Optional, List

from mapping_field.arithmetics import _ArithmeticMapFromFunction
from mapping_field.field import ExtElement
from mapping_field.mapping_field import MapElement, VarDict, CompositionFunction, OutputPromise, \
    always_validate_promises
from mapping_field.serializable import DefaultSerializable

IsCondition = OutputPromise("Condition")

class Condition(MapElement):
    pass

TrueCondition = None
FalseCondition = None

class BinaryCondition(Condition, DefaultSerializable):
    """
    An always True / False condition.
    """

    def __new__(cls, value: bool):
        if value:
            if TrueCondition is not None:
                return TrueCondition
        else:
            if FalseCondition is not None:
                return FalseCondition

        return super(BinaryCondition, cls).__new__(cls)

    def __init__(self, value: bool):
        super().__init__(variables=[])
        self.value = value
        self.add_promise(IsCondition)

    def to_string(self, entries: List[str]):
        return repr(self.value)

    def evaluate(self) -> Optional[ExtElement]:
        return 1 if self is TrueCondition else 0

    def invert(self) -> Optional[Condition]:
        return FalseCondition if (self is TrueCondition) else TrueCondition

TrueCondition  = BinaryCondition(True)
FalseCondition = BinaryCondition(False)


# <editor-fold desc="Not Condition">

@always_validate_promises
class _NotCondition(Condition, _ArithmeticMapFromFunction):

    def __init__(self):
        super().__init__('Not', lambda a: 1-a)
        self.add_promise(IsCondition)
        for v in self.vars:
            # TODO: Maybe switch directly to BoolVars?
            v.add_promise(IsCondition)

    def to_string(self, entries: List[str]):
        return f'~({entries[0]})'

    def _simplify_with_var_values2(self, var_dict: VarDict) -> Optional[MapElement]:
        entries = [var_dict.get(v,v) for v in self.vars]

        if not isinstance(entries[0], CompositionFunction):
            return super()._simplify_with_var_values2(var_dict)
        function = entries[0].function
        comp_entries = entries[0].entries
        if function == NotCondition:
            return comp_entries[0]

        return super()._simplify_with_var_values2(var_dict)

    def simplify(self):
        raise NotImplementedError('Delete this function')

NotCondition = _NotCondition()

def parameter_not_simplifier(var_dict: VarDict) -> Optional[MapElement]:
    entries = [var_dict[v] for v in NotCondition.vars]
    return entries[0].invert()

NotCondition.register_simplifier(parameter_not_simplifier)

MapElement.inversion = NotCondition

# </editor-fold>