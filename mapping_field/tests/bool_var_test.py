
from mapping_field.conditions import BinaryCondition, FalseCondition, TrueCondition
from mapping_field.log_utils.tree_loggers import TreeLogger, blue
from mapping_field.mapping_field import CompositeElement, MapElement
from mapping_field.promises import IsCondition
from mapping_field.ranged_condition import BoolVar, IntervalRange, RangeCondition

simplify_logger = TreeLogger(__name__)


def test_assignment():
    x = BoolVar("x")

    cond = x << 0

    assert cond is not TrueCondition
    assert cond is not FalseCondition

    cond0 = cond(0)
    cond1 = cond(1)

    assert cond0 is TrueCondition
    assert cond1 is FalseCondition


def test_union_assignment():
    x, y = BoolVar("x"), BoolVar("y")
    simplify_logger.tree.set_active(False)
    cond = (x << 0) & (y << 0)
    simplify_logger.tree.set_active(True)
    cond = cond({x: 0, y: 0})
    cond = cond.simplify2()
    assert cond is TrueCondition


def test_avoid_loop_in__two_bool_vars_simplifier():
    x = BoolVar("x")
    cond = TrueCondition & (x << 0)


def test_two_var_simplifier():

    x, y = BoolVar("x"), BoolVar("y")
    functions = [TrueCondition, FalseCondition]
    for v in [x, y]:
        for value in (0, 1):
            functions.append(v << value)
    for value_x, value_y in ((0, 0), (0, 1), (1, 0), (1, 2)):
        functions.append((x << value_x) & (y << value_y))
        functions.append((x << value_x) | (y << value_y))

    class TwoVar(CompositeElement):
        def __init__(self, function):
            super().__init__(operands=[x, y])
            self.promises.add_promise(IsCondition)
            self.function = function

        def _simplify_with_var_values2(self) -> MapElement | None:
            output_value = self.function({x:self.operands[0], y:self.operands[1]}).simplify2()
            if isinstance(output_value, BinaryCondition):
                return output_value
            return None

    for function in functions:
        simplify_logger.tree.reset()
        simplify_logger.log(f"Running test on {blue(function)}")
        two_var = TwoVar(function)
        assert function == two_var.simplify2(), f"Failed with the function {function}"


def test_two_var_simplifier2():

    x, y = BoolVar("x"), BoolVar("y")

    cond = (x << 0) & ((x << 0) | (y << 0))
    cond = cond.simplify2()
    assert cond == (x << 0)


def test_combine_bool_vars():
    x, y = BoolVar("x"), BoolVar("y")
    cond1 = (x << 0) & (y << 0)
    cond1 = cond1 | (y << 1)
    result = (x << 0) | (y << 1)
    assert cond1 == result


def test_1_to_3_complement():
    x, y = BoolVar("x"), BoolVar("y")

    cond1 = (x << 0) & (y << 0)
    cond2 = (x << 1) | (y << 1)
    result = cond1 | cond2

    assert result is TrueCondition


def test_bool_simplification():
    x = BoolVar("x")

    func = x * x
    # It equals to x, but there is no simplification for this (yet)
    func = func.simplify2()
    assert x != func
    assert str(func) == "(x*x)"

    # We can however simplify condition functions over booleans
    cond1 = (func << 1).simplify2()
    cond2 = x << 1
    assert cond1 == cond2


def test_two_bool_simplification():
    x, y = BoolVar("x"), BoolVar("y")

    cond1 = (x + y) << 0
    cond1 = cond1.simplify2()
    cond2 = (x << 0) & (y << 0)
    assert cond1 == cond2


def test_1_to_3_as_sum_complement():
    x, y = BoolVar("x"), BoolVar("y")

    cond1 = (x << 0) & (y << 0)
    cond2 = RangeCondition(x + y, IntervalRange[1, 2])
    result = cond1 | cond2

    assert result is TrueCondition


def test_four_bools():
    simplify_logger.tree.max_log_count = -1
    x1, x2, x3, x4 = BoolVar("x1"), BoolVar("x2"), BoolVar("x3"), BoolVar("x4")
    simplify_logger.tree.set_active(False)
    cond1 = (((x3 << 1) | (x4 << 0)) & (x2 << 1) & (x1 << 0)) | (((x1 << 1) | (x2 << 0)) & (x4 << 1) & (x3 << 0))
    cond2 = ((x1 << 1) | (x2 << 0)) & ((x3 << 1) | (x4 << 0))
    result = (x3 << 1) | (x4 << 0) | (x1 << 1) | (x2 << 0)

    simplify_logger.tree.set_active(True)
    cond = cond1 | cond2
    assert (
        str(cond1) == "[[(x3 = 0) & [(x2 = 0) | (x1 = 1)] & (x4 = 1)] | [(x1 = 0) & [(x4 = 0) | (x3 = 1)] & (x2 = 1)]]"
    )
    assert str(cond2) == "[[(x4 = 0) | (x3 = 1)] & [(x2 = 0) | (x1 = 1)]]"
    assert cond == result