from mapping_field.arithmetics import _Mult
from mapping_field.conditions import BinaryCondition, TrueCondition, FalseCondition, IntersectionCondition, \
    UnionCondition
from mapping_field.log_utils.tree_loggers import red
from mapping_field.mapping_field import Var, MapElement, simplifier_context, SimplifierOutput
from mapping_field.property_engines import is_condition, is_integral
from mapping_field.ranged_condition import in_range, IntervalRange, simplify_logger, RangeCondition
from mapping_field.utils.processors import ProcessFailureReason


def BoolVar(name: str) -> Var:
    return Var(name=name, output_properties={is_condition: True})


def is_bool_var(v: MapElement) -> bool:
    if not isinstance(v, Var):
        return False
    if is_condition.compute(v, simplifier_context):
        return True

    f_range = in_range.compute(v, simplifier_context)

    return (f_range is not None) and IntervalRange[0,1].contains(f_range) and is_integral.compute(v, simplifier_context)


@MapElement._simplifier.register_processor
def two_bool_vars_simplifier(elem: MapElement) -> SimplifierOutput:
    # TODO: make sure that I don't call has_promise for an element that I am trying to simplify, since it might
    #       call simplify inside it, and then we ar off to the infinite loop races.
    # if not is_condition.compute(elem, simplifier_context):
    if not is_condition.compute(elem, simplifier_context):
        return ProcessFailureReason("Not a Condition", trivial=True)
    # if len(var_dict) > 0:
    #     return ProcessFailureReason("Only applicable with no var_dict", trivial=True)
    if len(elem.vars) > 2 or (not all(is_bool_var(v) for v in elem.vars)):
        return ProcessFailureReason("Only applicable with at most 2 bool vars", trivial=True)

    if len(elem.vars) == 1:
        v = elem.vars[0]
        simplify_logger.log(f"Looking for simpler condition on {red(v)}")
        if elem in (v, (v << 0), (v << 1)):
            return None
        # TODO: The following two calls are problematics. They can generate composition function with 'elem'
        #       as the top function, so when we call simplify on it, it tries to simplify the top function
        #       by itself, which can loop back here.
        value0 = elem({v: 0}).simplify()
        value1 = elem({v: 1}).simplify()
        if not (isinstance(value0, BinaryCondition) and isinstance(value1, BinaryCondition)):
            simplify_logger.log(red(f"The values {value0}, {value1} should be binary."))
            return None
        if value0 is value1 is TrueCondition:
            return TrueCondition
        if value0 is value1 is FalseCondition:
            return FalseCondition
        return (v << 0) if value0 is TrueCondition else (v << 1)

    if len(elem.vars) == 2:
        x, y = elem.vars
        simplify_logger.log(f"Looking for simpler condition on {red(x)}, {red(y)}")
        assignments = [(0, 0), (0, 1), (1, 0), (1, 1)]
        values = [[elem({x: x0, y: y0}).simplify() for y0 in (0, 1)] for x0 in (0, 1)]
        if not all(isinstance(value, BinaryCondition) for value in values[0] + values[1]):
            values_str = ", ".join(str(value) for value in values[0] + values[1])
            simplify_logger.log(red(f"The values {values_str} should be binary."))
            return None

        count_true = sum(value is TrueCondition for value in values[0] + values[1])
        if count_true == 0:
            return FalseCondition
        if count_true == 1:
            for x0, y0 in assignments:
                if values[x0][y0] is TrueCondition:
                    result = IntersectionCondition([x << x0, y << y0], simplified=True)
                    # Don't call (x << x0) & (y << y0) since it automatically calls to simplify, which can
                    # cause an infinite loop
                    # TODO: should I split to structure simplified and promise simplified?
                    return result if elem != result else None
        if count_true == 2:
            if values[0][0] == values[1][1]:
                # TODO: add this when I know how to compare elements like (x+y = 1, x-y=0)
                return None
            v = y if (values[0][0] == values[1][0]) else x
            value = 0 if values[0][0] is TrueCondition else 1
            result = v << value
            return result
        if count_true == 3:
            for x0, y0 in assignments:
                if values[x0][y0] is FalseCondition:
                    result = UnionCondition([x << 1 - x0, y << 1 - y0], simplified=True)
                    return result if elem != result else None
        if count_true == 4:
            return TrueCondition

        # Should never get here...
        raise Exception("Learn how to count to 4...")

    return None


@_Mult.register_class_simplifier
def mult_binary_assignment_by_numbers(element: MapElement) -> SimplifierOutput:
    """
    change multiplications of (x << 1) * c into c * x for boolean variables x.
    """
    assert isinstance(element, _Mult)
    operands = element.operands
    value0 = operands[0].evaluate()
    value1 = operands[1].evaluate()
    if (value0 is None) + (value1 is None) != 1:
        return ProcessFailureReason("Exactly one of the factors must by a constant value", trivial=True)

    value = value0 or value1
    elem = operands[1] if value1 is None else operands[0]
    if isinstance(elem, RangeCondition) and is_bool_var(elem.function):
        if elem.range.is_point == 1 and value != 1:
            return value * elem.function
    return None
