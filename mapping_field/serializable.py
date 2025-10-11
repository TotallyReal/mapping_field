from abc import ABC, abstractmethod
import inspect
from typing import Dict, Type, List

_type = '_serialization_type'
_ref = '_serialization_reference'

class Serializable(ABC):
    """Interface for classes that can be saved to and loaded from dicts."""
    _subclasses = dict()
    _ref_to_objects = dict()
    _objects_to_refs = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Serializable._subclasses[cls.__name__] = cls
        # if cls != Serializable:
        #     assert 'from_dict' in cls.__dict__, f'Class {cls} must implement \'from_dict\''

    @abstractmethod
    def to_dict(self) -> Dict:
        """Convert object to a serializable dictionary."""
        raise Exception(f'Method \'to_dict\' is not implemented in {self.__class__}')

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict):
        """Reconstruct object from a dictionary."""
        raise Exception(f'Method \'from_dict\' is not implemented in {cls}')

class DefaultSerializable(Serializable):

    _name_conversion = 'serialized_fields'
    _serialized_fields: List[str] = []

    def __init_subclass__(cls, **kwargs):
        cls._serialized_fields = list(inspect.signature(cls.__init__).parameters)[1:]
        super().__init_subclass__(**kwargs)

    def serialization_name_conversion(self):
        return dict()

    @staticmethod
    def _dictify(element):
        if isinstance(element, (int, float, bool, str)):
            return element
        if isinstance(element, Serializable):
            return element.to_dict()
        if isinstance(element, list):
            return [DefaultSerializable._dictify(v) for v in element]
        if isinstance(element, tuple):
            return tuple([DefaultSerializable._dictify(v) for v in element])
        if isinstance(element, dict):
            return {DefaultSerializable._dictify(key) : DefaultSerializable._dictify(value)
                    for key, value in element.items()}

        raise Exception(f'Could not convert to dict: {element}')

    @staticmethod
    def _undictify(dict_rep):
        if isinstance(dict_rep, (int, float, bool, str)):
            return dict_rep
        if isinstance(dict_rep, list):
            return [DefaultSerializable._undictify(v) for v in dict_rep]
        if isinstance(dict_rep, tuple):
            return tuple([DefaultSerializable._undictify(v) for v in dict_rep])
        if isinstance(dict_rep, dict):
            if _type in dict_rep:
                value_cls = DefaultSerializable.get_class(dict_rep)
                return value_cls.from_dict(dict_rep)
            return {DefaultSerializable._undictify(key) : DefaultSerializable._undictify(value)
                    for key, value in dict_rep.items()}

        raise Exception(f'Could not rebuild the element from: {dict_rep}')

    def to_dict(self) -> Dict:
        serialized_dict = dict()
        conversion = self.serialization_name_conversion()
        for field_name in self.__class__._serialized_fields:
            value = getattr(self, conversion.get(field_name, field_name))
            # TODO: raise exception if value is not Serializable or "standard" (e.g. int, str, list, etc)?
            serialized_dict[field_name] = DefaultSerializable._dictify(value)

        serialized_dict[_type] = self.__class__.__name__
        serialized_dict[_ref]  = id(self)
        return serialized_dict

    @classmethod
    def from_dict(cls, dict_rep):
        ref = None
        if _ref in dict_rep:
            ref = dict_rep[_ref]
            if ref in Serializable._ref_to_objects:
                return Serializable._ref_to_objects[ref]

        parameters = dict()
        for key, value in dict_rep.items():
            if key in (_type, _ref):
                continue
            parameters[key] = DefaultSerializable._undictify(value)

        return cls(**parameters)

    @staticmethod
    def get_class(dict_rep) -> Type[Serializable]:
        if _type not in dict_rep:
            raise Exception(f'Dictionary representation must have a type.\n{dict_rep}')
        type_name = dict_rep[_type]
        if type_name not in Serializable._subclasses:
            raise Exception(f'Class type {type_name} is not registered as Serializable.')
        return Serializable._subclasses[type_name]