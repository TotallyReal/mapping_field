import pytest
import yaml

from mapping_field.mapping_field import Var, DefaultSerializable, NamedFunc, CompositionFunction, MapElement, MapElementConstant
from mapping_field.binary_expansion import BoolVar, BinaryExpansion
from mapping_field.conditional_function import ConditionalFunction
from mapping_field.conditions import ConditionUnion
from mapping_field.linear import Linear
from mapping_field.ranged_condition import RangeCondition, SingleAssignmentCondition
from mapping_field.serializable import Serializable


@pytest.fixture(autouse=True)
def reset_static_variables():
    Var.clear_vars()
    NamedFunc.clear_vars()

def process(elem: Serializable):

    orig_data = elem.to_dict()
    yaml_rep = yaml.dump(orig_data)
    result_data = yaml.safe_load(yaml_rep)

    cls = elem.__class__
    assert cls == DefaultSerializable.get_class(result_data)
    result = cls.from_dict(result_data)

    return result


def test_var_serialization():
    x = Var('x')
    assert x == process(x)

def test_named_func_serialization():
    x, y = Var('x'), Var('y')
    f = NamedFunc('f',[x,y])

    assert f == process(f)

def test_negative():
    x= Var('x')

    neg = -x
    serialization = neg.to_dict()
    cls = DefaultSerializable.get_class(serialization)
    assert cls == CompositionFunction
    g: CompositionFunction = cls.from_dict(serialization)
    assert g.function == MapElement.negation
    assert g.entries == [x]

def test_addition():
    x, y = Var('x'), Var('y')

    addition = x + y
    serialization = addition.to_dict()
    cls = DefaultSerializable.get_class(serialization)
    assert cls == CompositionFunction
    g: CompositionFunction = cls.from_dict(serialization)
    assert g.function == MapElement.addition
    assert g.entries == [x, y]

def test_binary_expansion():
    vv = [BoolVar(f'x_{i}') for i in range(4)]
    x = BinaryExpansion(vv)

    x == process(x)

def test_linear():
    x = Var('x')
    elem = Linear(5, x, 7)

    elem == process(elem)


def test_assignment_condition():
    x = Var('x')
    condition = SingleAssignmentCondition(x, 10)

    condition == process(condition)


def test_ranged_condition():
    x = Var('x')
    condition = (x < 10)
    condition = RangeCondition(x, (3,10))

    condition == process(condition)


def test_union_condition():
    xx = [Var(f'x_{i}') for i in range(3)]
    conditions = [(x < 10) for x in xx]

    condition: ConditionUnion = conditions[0] | conditions[1] | conditions[2]

    serialization = condition.to_dict()
    assert ConditionUnion == DefaultSerializable.get_class(serialization)
    result = ConditionUnion.from_dict(serialization)

    assert result.conditions == condition.conditions


def test_conditional_function():
    xx = [Var(f'x_{i}') for i in range(3)]

    func = ConditionalFunction([
        ( (xx[0] <= 5), MapElementConstant(1)),
        ( RangeCondition(xx[1], (3,10)), xx[1])
    ])

    serialization = func.to_dict()
    assert ConditionalFunction == DefaultSerializable.get_class(serialization)
    result = ConditionalFunction.from_dict(serialization)

    assert result.regions == func.regions
