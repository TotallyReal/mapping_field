import pytest
import logging
from typing import List, Tuple, Union, Set

from mapping_field.tree_loggers import TreeLogger
from mapping_field.binary_expansion import BoolVar, BinaryExpansion
from mapping_field.conditions import Condition, FalseCondition
from mapping_field.conditional_function import ConditionalFunction, ReLU
from mapping_field.linear import Linear
from mapping_field.mapping_field import MapElementConstant, MapElement, Var, NamedFunc
from mapping_field.ranged_condition import RangeCondition, SingleAssignmentCondition
from mapping_field.tests.conftest import debug_step, log_to_file, SIMPLE_FORMAT

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var.clear_vars()
    NamedFunc.clear_vars()


@pytest.fixture(autouse=True)
def reset_logs():
    TreeLogger.reset()

# <editor-fold desc=" ------------------------ Dummy objects ------------------------">

class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([])
        self.value = value

    def to_string(self, vars_str_list: List[str]):
        return f'DummyMap({self.value})'

class DummyCondition(Condition):
    def __init__(self, values: Union[int, Set[int]]=0, type: int=0):
        super().__init__([])
        self.values: Set[int] = set([values]) if isinstance(values, int) else values
        self.type = type

    def __repr__(self):
        return f'DummyCond_{self.type}({self.values})'

    def and_simpler(self, condition: Condition) -> Tuple['Condition', bool]:
        if isinstance(condition, DummyCondition) and self.type == condition.type:
            intersection = self.values.intersection(condition.values)
            return DummyCondition(intersection) if len(intersection) > 0 else FalseCondition, True
        return super().and_simpler(condition)

    def or_simpler(self, condition: Condition) -> Tuple['Condition', bool]:
        if isinstance(condition, DummyCondition) and self.type == condition.type:
            union = self.values.union(condition.values)
            return DummyCondition(union), True
        return super().or_simpler(condition)

    def _eq_simplified(self, other: Condition) -> bool:
        return (isinstance(other, DummyCondition) and
                self.type == other.type and
                len(self.values) == len(other.values) and
                all([v in other.values for v in self.values]))

# </editor-fold>


def test_equality_to_standard_function():
    dummies = [DummyCondition(i) for i in range(3)]
    cc = [MapElementConstant(i) for i in range(3)]

    cond_func = ConditionalFunction([
        (dummies[0], cc[0]),
        (dummies[1], cc[1]),
        (dummies[2], cc[2]),
    ])

    assert cond_func.evaluate() is None

    cond_func = ConditionalFunction([
        (dummies[0], cc[1]),
        (dummies[1], cc[1]),
        (dummies[2], cc[1]),
    ])

    assert cond_func.evaluate() == 1
    assert cond_func == 1

    dummy_map = DummyMap(0)

    cond_func = ConditionalFunction([
        (dummies[0], dummy_map),
        (dummies[1], dummy_map),
        (dummies[2], dummy_map),
    ])

    assert cond_func == dummy_map


def test_equality_to_conditional_function():

    dummies = [DummyCondition(i) for i in range(3)]
    cc = [MapElementConstant(i) for i in range(3)]


    cond_func1 = ConditionalFunction([
        (dummies[0], cc[0]),
        (dummies[1], cc[1]),
        (dummies[2], cc[2]),
    ])
    cond_func2 = ConditionalFunction([
        (dummies[0], cc[0]),
        (dummies[1], cc[1]),
        (dummies[2], cc[2]),
    ])

    assert cond_func1 == cond_func2

def test_combining_regions():
    x, y = BoolVar('x'), BoolVar('y')

    func = ConditionalFunction([
        ( (x << 0) & (y << 0) , MapElementConstant(0)),
        ( (x << 1) & (y << 0) , MapElementConstant(1)),
        ( (y << 1) , x),
    ])

    func = func.simplify2()
    assert func == x


def test_equality_region_wise():
    x = Var('x')

    cond_func = ConditionalFunction([
        (SingleAssignmentCondition(x, 0), MapElementConstant(0)),
        (SingleAssignmentCondition(x, 2), MapElementConstant(2)),
        (RangeCondition(x, (7, 17)), x),
    ])

    assert cond_func == x


def test_addition():
    dummies = [DummyCondition(i) for i in range(5)]

    cond_func1 = ConditionalFunction([
        (dummies[0] | dummies[1], MapElementConstant(0)),
        (dummies[2], MapElementConstant(10))
    ])

    cond_func2 = ConditionalFunction([
        (dummies[0], MapElementConstant(100)),
        (dummies[1] | dummies[2], MapElementConstant(200))
    ])

    cond_add = cond_func1 + cond_func2

    result = ConditionalFunction([
        (dummies[0], MapElementConstant(100)),
        (dummies[1], MapElementConstant(200)),
        (dummies[2], MapElementConstant(210))
    ])

    assert result == cond_add, f'could not match:\n{result}\n{cond_add}'


def test_addition_with_ranges():
    dummy_map = DummyMap(0)

    def ranged(low, high):
        return RangeCondition(dummy_map, (low, high))

    cond_func1 = ConditionalFunction([
        (ranged(0,10), MapElementConstant(0)),
        (ranged(10,30), MapElementConstant(10))
    ])

    cond_func2 = ConditionalFunction([
        (ranged(0,20), MapElementConstant(100)),
        (ranged(20,30), MapElementConstant(200))
    ])

    cond_add = cond_func1 + cond_func2

    result = ConditionalFunction([
        (ranged(10,20), MapElementConstant(110)),
        (ranged(0,10), MapElementConstant(100)),
        (ranged(20,30), MapElementConstant(210))
    ])

    assert  result == cond_add


def test_simplification():

    # combine regions with the same function
    dummy_cond = [DummyCondition(i) for i in range(5)]
    dummy_func = [DummyMap(i) for i in range(5)]

    cond_func = ConditionalFunction([
        (dummy_cond[0], dummy_func[0]),
        (dummy_cond[1], dummy_func[1]),
        (dummy_cond[2], dummy_func[0]),
    ])
    cond_func = cond_func.simplify2()

    simplified = ConditionalFunction([
        (dummy_cond[0] | dummy_cond[2], dummy_func[0]),
        (dummy_cond[1], dummy_func[1]),
    ])

    assert cond_func == simplified

    # Combine regions with assignemtns
    x = Var('x')
    xx = Linear.of(x)

    cond_func = ConditionalFunction([
        (RangeCondition(x, (0, 10)), xx + 3),
        (SingleAssignmentCondition(x, 10), MapElementConstant(13)),
    ])

    cond_func = cond_func.simplify2()

    assert cond_func == xx + 3


def test_linear_ranged_condition_subtraction():
    vv = [BoolVar(f'x_{i}') for i in range(4)]
    x = BinaryExpansion(vv)
    xx = Linear.of(x)

    v1 = ReLU(xx-7)
    v2 = ReLU(xx-8)
    v = v1 - v2
    v = v.simplify2()

    # TODO: improve union \ intersection of conditions

    assert v == x.coefficients[3]

    v = 8 * v
    u = xx - v

    result = BinaryExpansion(vv[:3])
    assert u == result


def test_general_assignment():
    x = Linear.of(Var('x'))

    func = ConditionalFunction([
        (x<0, -x),
        (x>=0, x + 7)
    ])

    condition1 = (func.where() == 10)
    condition2 = (x.where() == -10) | (x.where() == 3)
    assert condition1 == condition2