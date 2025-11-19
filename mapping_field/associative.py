from typing import List, Optional

from mapping_field.log_utils.tree_loggers import TreeLogger, red, green
from mapping_field.mapping_field import CompositeElement, MapElement

simplify_logger = TreeLogger(__name__)

class AssociativeListFunction(CompositeElement):

    trivial_element = None      # x @ trivial = x
    final_element   = None      # x @ final   = final

    @classmethod
    def unpack_list(cls, elements: List[MapElement]) -> List[MapElement]:
        elements = elements.copy()
        all_elements = []

        while len(elements) > 0:
            element = elements.pop(0)
            if isinstance(element, cls):
                elements.extend(element.operands)
                continue
            all_elements.append(element)
        return all_elements

    def __init__(self, operands: List[MapElement], simplified: bool = False):
        super().__init__(operands=self.__class__.unpack_list(operands), simplified=simplified)
        self._binary_special = False

    def _simplify_with_var_values2(self) -> Optional[MapElement]:
        if self._binary_special:
            simplify_logger.log("is binary special - avoid simplifying here.")
            return None

        cls = self.__class__

        final_operands = []
        operands = self.operands.copy()

        is_whole_simpler = False

        for operand in operands:

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
                # unpack list operand of the same type
                is_whole_simpler = True
                operands.extend(operand.operands)
                continue

            # Check if this new operand intersects in a special way with an existing operand.
            # Each time this loop repeats itself, the operands array's size must decrease by 1, so it cannot
            # continue forever.
            for existing_operand in final_operands:
                simplify_logger.log(
                    f"Trying to combine {red(operand)} with existing {red(existing_operand)}",
                )
                binary_op = cls([existing_operand, operand])
                binary_op._binary_special = True    # This will not loop back to be simplified here
                simplified_binary_op = binary_op._simplify2()

                if simplified_binary_op is not None:
                    simplify_logger.log(
                        f"Combined: {red(operand)} with {red(existing_operand)}  =>  {green(simplified_binary_op)}",
                    )
                    is_whole_simpler = True
                    final_operands = [cond for cond in final_operands if (cond is not existing_operand)]
                    operands.append(simplified_binary_op)
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