import pytest
import random

# from ..field import FieldElement
from mapping_field import Var, Func, MapElementFromFunction

#
# def plus_one(elem):
#     return elem + 1
#
#
# def times_two(elem):
#     return elem * 2


def test_associativity():
    x = Var('x')
    y = Var('y')
    z = Var('z')

    f = Func('f')(x)

    # lhs = y*f(x)
    # rhs = y*x - f(1/x)
    print('\n-------------------')
    g=y*x
    print(g)
    h = g(x=1/x)
    print(h)
    # print(lhs)
    # print(rhs)
    # print(rhs(x=1/x))
    # print('^^^^')


    #
    # Creates a two variable function called 'f'
    # f = Func('f')(x, y)
    # print('\n-------------------')
    # print(f)
    # print(f(1, f(z, 2)))
    #
    # g = MapElementFromFunction(name='sub', function = lambda a, b: a-b)
    # print(g(1, 2))
    #
    # print(g(1, 2) + 1 + y*z*0)

    # g = f(x, x)
    # print('\n-------------------')
    # print(g)
    #
    # h = g(x=f)
    # print('\n-------------------')
    # print(h)
    # print('\n-------------------')
    # print(f(f(x, x),h))
    # print(f(f(1, 2), y))
    # The associativity formula for 'f'
    # formula = f(f(x, y), z) - f(x, f(y, z))
    # print(formula)
    assert False


    addition = lambda a, b: a + b           # A python function of two variables (addition)
    multiplication = lambda a, b: a * b     # A python function of two variables (multiplication)



    addition_formula = formula(f=addition)
    assert addition_formula(1, 2, 3) == 0


    multiplication_formula = formula(f=multiplication)
    assert addition_formula(1, 2, 3) == 0



    # f = UniMapping(plus_one, x)         # The function x + 1
    # g = UniMapping(times_two, y)        # The function y * 2
    # print(g(f(FieldElement(3))).evaluate())
    #
    # F_of_X = UniFunctionMap('F', X)
    # assert False