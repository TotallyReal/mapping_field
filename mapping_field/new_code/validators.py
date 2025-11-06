from typing import Callable, Generic, List, Optional, Type, TypeVar

T = TypeVar('T', bound=object)
Context = TypeVar('Context', bound=object)

Validator = Callable[[T], bool]
ContextValidator = Callable[[T, Context], bool]

class MultiValidator(Generic[T, Context]):
    """
    A new validator which is the 'or' operation on a collection of other validators.
    """

    def __init__(self, name: Optional[str] = None, context: Optional[Context] = None):
        self.name = name or self.__class__.__name__
        self.validators: List[Validator[T]] = []
        self.context_validators: List[ContextValidator[T, Context]] = []
        self.direct = ValidatorByClassOrObject()
        self.register_validator(self.direct)
        self.context = context

    def __repr__(self):
        return self.name

    def __call__(self, value: T) -> bool:
        return self.validate(value)

    def validate(self, value: T) -> bool:
        if self.context is not None:
            for validator in self.context_validators:
                if validator(value, self.context):
                    return True
        for validator in self.validators:
            if validator(value):
                return True
        return False

    def register_validator(self, validator: Validator[T]):
        self.validators.append(validator)

    def register_context_validator(self, validator: ContextValidator[T, Context]):
        self.context_validators.append(validator)

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