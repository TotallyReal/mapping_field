from typing import Callable, Generic, List, Optional, Type, TypeVar

T = TypeVar('T', bound=object)

Validator = Callable[[T], bool]

class MultiValidator(Generic[T]):
    """
    A new validator which is the 'or' operation on a collection of other validators.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.validators: List[Validator[T]] = []
        self.direct = ValidatorByClassOrObject()
        self.register_validator(self.direct)

    def __repr__(self):
        return self.name

    def __call__(self, value: T) -> bool:
        return self.validate(value)

    def validate(self, value: T) -> bool:
        for validator in self.validators:
            if validator(value):
                return True
        return False

    def register_validator(self, validator: Validator[T]):
        self.validators.append(validator)

class ValidatorByClassOrObject(Generic[T]):

    def __init__(self):
        self._validated_classes = []
        self._validated_elements = []

    def register_class(self, cls_type: Type[T]):
        self._validated_classes.append(cls_type)

    def register_object(self, element: T):
        self._validated_elements.append(element)

    def __call__(self, value: T) -> bool:
        return (value.__class__ in self._validated_classes) or (value in self._validated_elements)