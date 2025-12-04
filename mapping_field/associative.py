from collections import deque

from mapping_field.log_utils.tree_loggers import TreeLogger, green, red
from mapping_field.mapping_field import (
    CompositeElement, MapElement, OutputProperties, SimplifierOutput, Var, convert_to_map,
    simplifier_context,
)
from mapping_field.utils.processors import ProcessFailureReason

simplify_logger = TreeLogger(__name__)

class AssociativeListFunction(CompositeElement):
    # TODO: add tests

    def __init_subclass__(cls, binary_class: type[CompositeElement] | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if binary_class is not None:
            def _to_multi_conversion(element: MapElement) -> SimplifierOutput:
                assert isinstance(element, binary_class)
                if AssociativeListFunction.is_binary(element):
                    return ProcessFailureReason("Is a binary construct.", trivial = True)

                multi_version = cls(operands=element.operands, output_properties=simplifier_context.get_user_properties(element))
                return multi_version

            binary_class.register_class_simplifier(_to_multi_conversion)

        cls.binary_class = binary_class or cls


    trivial_element = None      # x @ trivial = x
    final_element   = None      # x @ final   = final
    op_symbol       = "@"
    left_bracket    = "("
    right_bracket   = ")"
    unpack_single   = True      # TODO: this is just for ConditionalFunction

    binary_constructs: set[int] = set()

    def __eq__(self, other: MapElement) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if len(self.operands) != len(other.operands):
            return False

        for operand in self.operands:
            if operand not in other.operands:
                return False

        return True

    __hash__ = MapElement.__hash__

    @classmethod
    def is_trivial(cls, element: MapElement) -> bool:
        return element is cls.trivial_element

    @classmethod
    def is_binary(cls, element: MapElement) -> bool:
        return id(element) in cls.binary_constructs

    @classmethod
    def unpack_list(cls, elements: list[MapElement]) -> list[MapElement]:
        elements = elements.copy()
        all_elements = []

        while len(elements) > 0:
            element = convert_to_map(elements.pop(0))
            if isinstance(element, cls):
                if len(simplifier_context.get_user_properties(element)) == 0:
                    elements.extend(element.operands)
                continue
            all_elements.append(element)
        return all_elements

    def __init__(self, operands: list[MapElement], simplified: bool = False,
            output_properties: OutputProperties | None = None):
        super().__init__(operands=self.__class__.unpack_list(operands), simplified=simplified, output_properties=output_properties)

    def to_string(self, vars_to_str: dict[Var, str]):
        cls = self.__class__
        op_symbol = cls.op_symbol
        if cls.is_binary(self):
            op_symbol = op_symbol * 2
        op_symbol = f" {op_symbol} "
        operands_str = op_symbol.join(operand.to_string(vars_to_str) for operand in self.operands)
        return f"{cls.left_bracket}{operands_str}{cls.right_bracket}"

    def _simplify_with_var_values(self) -> MapElement | None:
        if self.__class__.is_binary(self):
            simplify_logger.log("is binary special - avoid simplifying here.")
            return None

        cls = self.__class__

        final_operands = []
        queue = deque(self.operands)

        is_whole_simpler = False

        while queue:
            operand = queue.popleft()

            simplified_condition = operand._simplify()
            if simplified_condition is not None:
                is_whole_simpler = True
            operand = simplified_condition or operand

            if operand is cls.final_element:
                simplify_logger.log(f"Contains final element {green(operand)}")
                return cls.final_element

            if cls.is_trivial(operand):
                simplify_logger.log(f"Removing trivial element {green(operand)}")
                is_whole_simpler = True
                continue

            if isinstance(operand, cls):
                if len(simplifier_context.get_user_properties(operand)) == 0:
                    # unpack list operand of the same type
                    simplify_logger.log(f"Unpacking {green(operand)}")
                    is_whole_simpler = True
                    queue.extendleft(reversed(operand.operands))
                    continue

            # Check if this new operand intersects in a special way with an existing operand.
            # Each time this loop repeats itself, the operands array's size must decrease by 1, so it cannot
            # continue forever.
            for existing_idx, existing_operand in enumerate(final_operands):
                simplify_logger.log(
                    f"Trying to combine {red(operand)} with existing {red(existing_operand)}",
                )

                user_properties = {}
                if len(final_operands) == 1 and len(queue) == 0:
                    user_properties = simplifier_context.get_user_properties(self)
                binary_op = cls.binary_class([existing_operand, operand], output_properties=user_properties)

                AssociativeListFunction.binary_constructs.add(id(binary_op))
                simplified_binary_op = binary_op._simplify()
                AssociativeListFunction.binary_constructs.remove(id(binary_op))

                if simplified_binary_op is not None:
                    if cls.binary_class != cls and isinstance(simplified_binary_op, cls.binary_class):
                        binary_op = cls(operands=simplified_binary_op.operands, output_properties=simplifier_context.get_user_properties(simplified_binary_op))
                        simplified_binary_op = binary_op
                    simplify_logger.log(
                        f"Combined: {red(operand)} with {red(existing_operand)}  =>  {green(simplified_binary_op)}",
                    )
                    is_whole_simpler = True
                    final_operands.pop(existing_idx)
                    queue.appendleft(simplified_binary_op)
                    break
            else:
                # new operand cannot be combined in a special way
                final_operands.append(operand)

        if len(final_operands) == 0:
            return cls.trivial_element

        if cls.unpack_single and len(final_operands) == 1:
            return final_operands[0]

        if is_whole_simpler:
            return cls(final_operands)

        return None


def sort_key(element: MapElement):
    # Comparison:
    #   1. First by number of variables
    #   2. If have the same number of variables, compare by their names (sorted)
    #   3. If have the exact same variables, just compare by the string representation of the functions.
    return (element.num_vars, sorted([v.name for v in element.vars]), str(element))



def _sorted_commutative_simplifier(element: MapElement) -> SimplifierOutput:
    """
            x4 @ x1 @ x3 @ x2   =>  x1 @ x2 @ x3 @ x4
    """
    assert isinstance(element, CompositeElement)
    operands = element.operands
    sorted_operands = sorted(operands, key=lambda operand: sort_key(operand))
    if all(op1 is op2 for op1, op2 in zip(sorted_operands, operands)):
        return ProcessFailureReason('No need to sort', trivial=True)
    return element.copy_with_operands(sorted_operands)

