import collections
import logging
import weakref
from typing import Generic, TypeVar, Iterator, Iterable

logger = logging.getLogger(__name__)

K = TypeVar("K")  # key type
V = TypeVar("V")  # value type

class GenericWeakKeyDictionary(Generic[K, V]):
    """
    A wrapper class for WeakKeyDictionary with type annotations
    """
    def __init__(self) -> None:
        self._data: weakref.WeakKeyDictionary[K, V] = weakref.WeakKeyDictionary()

    def validate(self, key: K):
        items = list(self._data.items())
        hashes = [hash(key) for key, _ in items]
        counters = collections.Counter(hashes)
        for key, count in counters.items():
            if count > 1:
                print(f'{key} has {count} items')
                indices = [i for i, hash_value in enumerate(hashes) if hash_value == key]
                corresponding_items = [items[i] for i in indices]
                print(corresponding_items)

        assert len(set(hashes)) == len(hashes), 'hash appears twice?!'

    def __setitem__(self, key: K, value: V) -> None:
        # logger.info(f"Setting {key} to {value}")
        # self.validate(key)
        flag = key in self._data
        self._data[key] = value
        # self.validate(key)

    def __getitem__(self, key: K) -> V:
        # logger.info(f"Getting {key}")
        # self.validate(key)
        return self._data[key]

    def __delitem__(self, key: K) -> None:
        del self._data[key]

    def __contains__(self, key: K) -> bool:
        return key in self._data

    def get(self, key: K, default: V | None = None) -> V | None:
        return self._data.get(key, default)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def keys(self) -> Iterable[K]:
        return self._data.keys()

    def values(self) -> Iterable[V]:
        return self._data.values()

    def items(self) -> Iterable[tuple[K, V]]:
        return self._data.items()

    def __repr__(self) -> str:
        return f"TypedWeakKeyDict({dict(self._data)})"