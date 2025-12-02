import random
import uuid

from typing import Any

from mapping_field.utils.weakref import GenericWeakKeyDictionary


class Key:
    def __init__(self, key: int):
        self.key = key
        self._hash = uuid.uuid4().int

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Key) and self.key == other.key

def test_keys():
    weak_dict = GenericWeakKeyDictionary[Key, bool]()
    for _ in range(10):
        keys = [Key(i) for i in range(10)]
        for _ in range(1000):
            key = random.choice(keys)
            weak_dict[key] = True
            weak_dict.validate(key)