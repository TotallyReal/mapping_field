from mapping_field import Var, DefaultSerializable, NamedFunc, MapElementFromFunction, CompositionFunction, MapElement
from mapping_field.binary_expansion import BoolVar, BinaryExpansion
from mapping_field.linear import Linear


def test_var_serialization():
    x = Var('x')
    serialization = x.to_dict()
    cls = DefaultSerializable.get_class(serialization)
    assert cls == Var
    y = cls.from_dict(serialization)
    assert x == y

def test_named_func_serialization():
    x, y = Var('x'), Var('y')
    f = NamedFunc('f',[x,y])
    serialization = f.to_dict()
    cls = DefaultSerializable.get_class(serialization)
    assert cls == NamedFunc
    g = cls.from_dict(serialization)
    assert f == g

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

    serialization = x.to_dict()
    cls = DefaultSerializable.get_class(serialization)
    assert cls == BinaryExpansion
    y: BinaryExpansion = cls.from_dict(serialization)
    assert x == y

def test_linear():
    x = Var('x')
    elem = Linear(5, x, 7)

    serialization = elem.to_dict()
    cls = DefaultSerializable.get_class(serialization)
    assert cls == Linear
    elem2: Linear = cls.from_dict(serialization)
    assert elem == elem2
