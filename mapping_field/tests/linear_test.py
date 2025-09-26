from typing import List

from mapping_field.linear import Linear
from mapping_field.conditions import RangeCondition, ConditionalFunction, ReLU
from mapping_field.mapping_field import MapElementConstant, MapElement

class DummyMap(MapElement):
    def __init__(self, value=0):
        super().__init__([])
        self.value = value

    def to_string(self, vars_str_list: List[str]):
        return f'DummyMap({self.value})'

    def __eq__(self, other):
        return isinstance(other, DummyMap) and other.value == self.value

def test_linear_generation():

    dummy = DummyMap(0)
    linear_dummy = Linear.of(dummy)

    func = 5*linear_dummy
    result = Linear(5, dummy, 0)
    assert func == result

    func = linear_dummy + 7
    result = Linear(1, dummy, 7)
    assert func == result

    func = 5*linear_dummy + 7
    result = Linear(5, dummy, 7)
    assert func == result

    func = 0*linear_dummy + 7
    func = func.simplify()
    result = 7
    assert func == result

def test_linear_arithmetic():
    dummy = Linear.of(DummyMap(0))

    func1 = 5*dummy + 3
    func2 = 11*dummy + 7

    func = func1 + func2
    result = 16*dummy + 10
    assert func == result

    func = func1 - func2
    result = -6*dummy - 4
    assert func == result


def test_linear_ranged_condition():
    dummy = DummyMap(0)
    func = 2*Linear.of(dummy) + 3
    condition = RangeCondition(func, (5,15))

    condition = condition.simplify()
    result = RangeCondition(dummy, (1,6))

    assert condition == result