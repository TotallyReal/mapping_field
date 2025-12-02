
from mapping_field.conditions import FalseCondition, TrueCondition
from mapping_field.mapping_field import MapElement, OutputProperties, Var
from mapping_field.property_engines import is_condition


class DummyMap(MapElement):
    def __init__(self, value=0, output_properties: OutputProperties | None = None):
        self.value = value
        super().__init__([], f"DummyMap({value})", output_properties=output_properties)

    def to_string(self, vars_to_str: dict[Var, str]):
        return f"DummyMap({self.value})"

    def __eq__(self, other):
        return isinstance(other, DummyMap) and other.value == self.value

    __hash__ = MapElement.__hash__


class DummyCondition(MapElement):
    def __init__(self, type: int = 0, values: int | set[int] = 0):
        self.values: set[int] = set([values]) if isinstance(values, int) else values
        self.type = type
        super().__init__([], output_properties={is_condition: True})

    def to_string(self, vars_to_str: dict[Var, str]):
        return f"DummyCond_{self.type}({self.values})"

    def and_(self, condition: MapElement) -> MapElement | None:
        if isinstance(condition, DummyCondition) and self.type == condition.type:
            intersection = self.values.intersection(condition.values)
            return DummyCondition(values=intersection, type=self.type) if len(intersection) > 0 else FalseCondition
        return None

    def or_(self, condition: MapElement) -> MapElement | None:
        if isinstance(condition, DummyCondition) and self.type == condition.type:
            union = self.values.union(condition.values)
            return DummyCondition(values=union, type=self.type)
        return None

    def __eq__(self, other: MapElement) -> bool:
        return (
            isinstance(other, DummyCondition)
            and self.type == other.type
            and len(self.values) == len(other.values)
            and all([v in other.values for v in self.values])
        )

    __hash__ = MapElement.__hash__


class DummyConditionOn(MapElement):
    def __init__(self, set_size: int = 1, values: int | set[int] = 0):
        self.values: set[int] = set([values]) if isinstance(values, int) else values
        assert all(0<=value<set_size for value in self.values)
        self.set_size = set_size
        super().__init__([], output_properties={is_condition: True})

    def to_string(self, vars_to_str: dict[Var, str]):
        return f"DummyCondOn_{self.set_size}({self.values})"

    def and_(self, condition: MapElement) -> MapElement | None:
        if isinstance(condition, DummyConditionOn) and self.set_size == condition.set_size:
            intersection = self.values.intersection(condition.values)
            return DummyConditionOn(values=intersection, set_size=self.set_size) if len(intersection) > 0 else FalseCondition
        return None

    def or_(self, condition: MapElement) -> MapElement | None:
        if isinstance(condition, DummyConditionOn) and self.set_size == condition.set_size:
            union = self.values.union(condition.values)
            return DummyConditionOn(values=union, set_size=self.set_size) if len(union) < self.set_size else TrueCondition
        return None

    def __eq__(self, other: MapElement) -> bool:
        return (
            isinstance(other, DummyConditionOn)
            and self.set_size == other.set_size
            and len(self.values) == len(other.values)
            and all([v in other.values for v in self.values])
        )

    __hash__ = MapElement.__hash__
