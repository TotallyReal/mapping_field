from typing import List, Tuple, Union, Set

from mapping_field.binary_expansion import BoolVar, BinaryExpansion
from mapping_field.conditions import Condition, FalseCondition
from mapping_field.conditional_function import ConditionalFunction, ReLU
from mapping_field.linear import Linear
from mapping_field.mapping_field import MapElementConstant, MapElement
from mapping_field.ranged_condition import RangeCondition


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

def test_op_conditional_functions():
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

def test_op_conditional_functions_ranges():
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
    u = ConditionalFunction.always(xx) - v
    u = u.simplify2()

    result = BinaryExpansion(vv[:3])
    assert u == result