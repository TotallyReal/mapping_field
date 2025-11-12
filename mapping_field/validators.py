from typing import Callable, Generic, List, Optional, Type, TypeVar

T = TypeVar("T", bound=object)
Context = TypeVar("Context", bound=object)

Validator = Callable[[T], Optional[bool]]
ContextValidator = Callable[[T, Context], Optional[bool]]


class MultiValidator(Generic[T, Context]):
    """
    A new validator which is the output from a collection of validators, namely at least one needs to validate
    True\False to get this output
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

    def __call__(self, value: T) -> Optional[bool]:
        return self.validate(value)

    def validate(self, value: T) -> Optional[bool]:
        if self.context is not None:
            for validator in self.context_validators:
                validation = validator(value, self.context)
                if validation is not None:
                    return validation
        for validator in self.validators:
            validation = validator(value)
            if validation is not None:
                return validation
        return None

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

    def __call__(self, value: T) -> Optional[bool]:
        if (value.__class__ in self._validated_classes) or (value in self._validated_elements):
            return True
        return None
