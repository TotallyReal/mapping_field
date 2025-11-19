from collections import deque
from typing import Dict, List, Optional, Union

from mapping_field.log_utils.tree_loggers import TreeLogger, green, red
from mapping_field.mapping_field import CompositeElement, MapElement, Var, convert_to_map
from mapping_field.utils.processors import ProcessFailureReason

simplify_logger = TreeLogger(__name__)

class AssociativeListFunction(CompositeElement):
    # TODO: add tests

    trivial_element = None      # x @ trivial = x
    final_element   = None      # x @ final   = final
    op_symbol       = "@"
    left_bracket    = "("
    right_bracket   = ")"

    @classmethod
    def is_trivial(cls, element: MapElement) -> bool:
        return element is cls.trivial_element

    @classmethod
    def unpack_list(cls, elements: List[MapElement]) -> List[MapElement]:
        elements = elements.copy()
        all_elements = []

        while len(elements) > 0:
            element = convert_to_map(elements.pop(0))
            if isinstance(element, cls):
                for promise in element.promises.output_promises():
                    if promise not in cls.auto_promises:
                        all_elements.append(element)
                        break
                else:
                    elements.extend(element.operands)
                continue
            all_elements.append(element)
        return all_elements

    def __init__(self, operands: List[MapElement], simplified: bool = False):
        super().__init__(operands=self.__class__.unpack_list(operands), simplified=simplified)
        self._binary_special = False

    def to_string(self, vars_to_str: Dict[Var, str]):
        cls = self.__class__
        op_symbol = cls.op_symbol
        if self._binary_special:
            op_symbol = op_symbol * 2
        op_symbol = f" {op_symbol} "
        operands_str = op_symbol.join(operand.to_string(vars_to_str) for operand in self.operands)
        return f"{cls.left_bracket}{operands_str}{cls.right_bracket}"

    def _simplify_with_var_values2(self) -> Optional[MapElement]:
        if self._binary_special:
            simplify_logger.log("is binary special - avoid simplifying here.")
            return None

        cls = self.__class__

        final_operands = []
        queue = deque(self.operands)

        is_whole_simpler = False

        while queue:
            operand = queue.popleft()

            simplified_condition = operand._simplify2()
            if simplified_condition is not None:
                is_whole_simpler = True
            operand = simplified_condition or operand

            if operand is cls.final_element:
                return cls.final_element

            if operand is cls.trivial_element:
                is_whole_simpler = True
                continue

            if isinstance(operand, cls):
                for promise in operand.promises.output_promises():
                    if promise not in cls.auto_promises:
                        break
                else:
                    # unpack list operand of the same type
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
                binary_op = cls([existing_operand, operand])
                if len(final_operands) == 1 and len(queue) == 0:
                    binary_op.promises = self.promises.copy()
                binary_op._binary_special = True    # This will not loop back to be simplified here
                simplified_binary_op = binary_op._simplify2()

                if simplified_binary_op is not None:
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

        if len(final_operands) == 1:
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


def _sorted_commutative_simplifier(element: MapElement) -> Optional[Union[ProcessFailureReason,MapElement]]:
    assert isinstance(element, CompositeElement)
    operands = element.operands
    sorted_operands = sorted(operands, key=lambda operand: sort_key(operand))
    if all(op1 is op2 for op1, op2 in zip(sorted_operands, operands)):
        return ProcessFailureReason('No need to sort', trivial=True)
    return element.copy_with_operands(sorted_operands)


