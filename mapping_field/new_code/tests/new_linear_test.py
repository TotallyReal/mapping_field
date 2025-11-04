import pytest
from typing import List

from mapping_field.log_utils.tree_loggers import TreeLogger
from mapping_field.new_code.mapping_field import MapElement, Var
from mapping_field.new_code.linear import Linear


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var.clear_vars()

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
    func = func.simplify2()
    result = 7
    assert func == result

def test_linear_unpacking():
    dummy = DummyMap(0)

    elem1 = Linear(3, Linear(4, dummy, 5), 6)
    elem2 = Linear(12, dummy, 21)
    assert elem1 == elem2

def single_addition(func1: MapElement, func2: MapElement, addition: MapElement):
    TreeLogger._paused = False
    result = func1 + func2
    assert result == addition
    assert result - func1 == func2
    assert result - func2 == func1

def test_linear_addition():
    TreeLogger._paused = True
    dummy = DummyMap(0)
    lin_dummy = Linear.of(dummy)

    # Add constant
    single_addition(5*lin_dummy + 3, 3, 5*lin_dummy + 6)

    # Add elem
    single_addition(5*lin_dummy + 3, dummy, 6*lin_dummy + 3)

    # Add linear
    single_addition(5*lin_dummy + 3, 11*lin_dummy + 7, 16*lin_dummy + 10)

    # Add unpacked linear
    # single_addition(5*lin_dummy + 3, 11*dummy     + 7, 16*lin_dummy + 10)


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
#
# def test_conversion_to_linear():
#     dummy = DummyMap(0)
#
#     # linear of general functions
#     lin_dummy = Linear(1, dummy, 0)
#     assert Linear.of(dummy) == lin_dummy
#
#     # linear of linear functions: same function (not just equals)
#     lin_dummy = Linear(3, dummy, 4)
#     assert Linear.of(lin_dummy) is lin_dummy
#
#     arith_dummy = 3 * dummy + 4
#     assert Linear.of(arith_dummy) == lin_dummy
#
#     arith_dummy = 4 + dummy * 3
#     assert Linear.of(arith_dummy) == lin_dummy
#
# # ============================== ranged condition ==============================
#
# def test_linear_ranged_condition():
#     dummy = DummyMap(0)
#     linear_dummy = Linear.of(dummy)
#
#     def ranged_condition(a, b, low, high) -> Condition:
#         func = a*linear_dummy + b
#         condition = RangeCondition(func, (low, high))
#         condition = condition.simplify()
#         return condition
#
#     condition = ranged_condition(a=2, b=3, low=5, high=15)
#     result = RangeCondition(dummy, (1,6))
#     assert condition == result
#
#     # test negative coefficient
#     condition = ranged_condition(a=-2, b=3, low=5, high=15)
#     result = RangeCondition(dummy, (-5,0))
#     assert condition == result
#
#     # test zero coefficient - False
#     condition = ranged_condition(a=0, b=3, low=5, high=15)
#     result = FalseCondition
#     assert condition == result
#
#     # test zero coefficient - True
#     condition = ranged_condition(a=0, b=7, low=5, high=15)
#     result = TrueCondition
#     assert condition == result
#
#
# def test_assignment_condition():
#     Var.clear_vars()
#     x, y, z = Var('x'), Var('y'), Var('z')
#
#     # disjoint
#     cond1 = SingleAssignmentCondition.from_assignment_dict({x:1, y:2})
#     cond2 = SingleAssignmentCondition.from_assignment_dict({z:3})
#     prod = cond1 & cond2
#     result = SingleAssignmentCondition.from_assignment_dict({x:1, y:2, z:3})
#     assert prod == result
#
#     # valid intersection
#     cond1 = SingleAssignmentCondition.from_assignment_dict({x:1, y:2})
#     cond2 = SingleAssignmentCondition.from_assignment_dict({y:2, z:3})
#     prod = cond1 & cond2
#     result = SingleAssignmentCondition.from_assignment_dict({x:1, y:2, z:3})
#     assert prod == result
#
#     # invalid intersection
#     cond1 = SingleAssignmentCondition.from_assignment_dict({x:1, y:2})
#     cond2 = SingleAssignmentCondition.from_assignment_dict({y:5, z:3})
#     prod = cond1 & cond2
#     result = FalseCondition
#     assert prod == result
#
#
# # def test_assignment_from_range():
# #     x = BinaryExpansion.generate('x', 4)
# #
# #     condition = RangeCondition(x, (5,6))
# #     condition = condition.simplify()
# #     result = AssignmentCondition({x: 5})
# #     assert condition == result
# #
# #     linear_func = Linear(5, x, 2)
# #     condition = RangeCondition(linear_func, (7,12))
# #     condition = condition.simplify()
# #     result = AssignmentCondition({x: 1})
# #     assert condition == result
#
# def test_linear_shift():
#     v1 = BoolVar('v1')
#     v2 = BoolVar('v2')
#
#     x1 = Linear.of(BinaryExpansion([v1, v2, 0, 1, 1]))
#     x2 = Linear.of(BinaryExpansion([0, 0, v1, v2, 0, 1, 1]))
#     x3 = Linear.of(BinaryExpansion([0, 0, 0, 0, v1, v2, 0, 1, 1]))
#
#     # The result of the arithmetics operations are simplified automatically
#
#     assert x1*4 == x2.simplify2()
#     assert x1*16 == x3.simplify2()
#     assert x2*4 == x3.simplify2()
#
#
# def addition_test(x, y, x_plus_y):
#     addition = x + y
#     assert addition == x_plus_y
#     difference = x_plus_y - x
#     assert difference.simplify2() == y.simplify2()
#     difference = x_plus_y - y
#     assert difference.simplify2() == x.simplify2()
#
# def test_linear_addition_of_binary_expansion():
#     v = [BoolVar(f'v_{i}') for i in range(4)]
#
#     x = Linear.of(BinaryExpansion([v[0], 0, v[2]]))
#     y = Linear.of(BinaryExpansion([0, v[0], 0, v[2]]))
#     result = 5*y
#     addition_test(6 * x, 2 * y, result)
#
#
#     y = Linear.of(BinaryExpansion([v[1], 1, v[3]]))
#
#     x = 3*x + 1
#     y = 6*y + 5
#     result = Linear.of(BinaryExpansion([v[0], v[1], v[2], v[3]]))
#     result = 3 * result + 18
#     addition_test(x, y, result)
#
# def test_me():
#     vv = [BoolVar(f'v_{i}') for i in range(4)]
#     x = BinaryExpansion(vv)
#     xx = Linear.of(x)
#
#     cond1 = xx - 7 >= 0
#     cond2 = xx - 8 >= 0
#     assert cond1 & cond2 == cond2
#
#     cond1 = xx - 7 >= 0
#     cond2 = xx - 8 < 0
#     prod = cond1 & cond2
#     prod = prod.simplify()
#     assert prod == SingleAssignmentCondition.from_assignment_dict({vv[0]:1, vv[1]:1, vv[2]: 1, vv[3]:0})
#
#     cond1 = xx - 7 < 0
#     cond2 = xx - 8 >= 0
#     prod = cond1 & cond2
#     prod = prod.simplify()
#     assert prod == FalseCondition
#
#     cond1 = xx - 7 < 0
#     cond2 = xx - 8 < 0
#     prod = cond1 & cond2
#     prod = prod.simplify()
#     assert prod == cond1
#
#
# def test_assignment_range_condition():
#     vv = [BoolVar(f'x_{i}') for i in range(4)]
#     x = BinaryExpansion(vv)
#     xx = Linear.of(x)
#
#     y = BinaryExpansion(vv[:3])
#     yy = Linear.of(y)
#
#     cond1 = (xx-7<0).simplify()
#     cond2 = (yy-7<0).simplify()
#     cond2 = cond2 & SingleAssignmentCondition(vv[3], 0)
#     assert cond1 == cond2
#
#
# def test_general_assignment():
#     dummy = DummyMap()
#     lin_dummy = -3*Linear.of(dummy)+2
#
#     condition1 = (lin_dummy.where() == 17)
#     condition2 = (dummy.where() == -5)
#     assert condition1 == condition2
#
