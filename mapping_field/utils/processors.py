import dataclasses
import weakref

from typing import Callable, Generic, TypeVar

from colorama import Back, init

from mapping_field.log_utils.tree_loggers import TreeAction, TreeLogger, cyan, green, magenta, red

init(autoreset=True)
logger = TreeLogger(__name__)

K = TypeVar("K")  # key type (any object)
V = TypeVar("V")  # value type (context)

class WeakContextDictionary(Generic[K, V]):
    """
    Dictionary-like object mapping id(obj) -> context.
    Automatically removes entries when the object is garbage-collected.
    Works for any object, even unhashable.
    """

    def __init__(self):
        self._data: dict[int, V] = {}                       # id(obj) -> context
        self._finalizers: dict[int, weakref.finalize] = {}  # id(obj) -> finalizer

    def __setitem__(self, obj: K, context: V) -> None:
        oid = id(obj)
        self._data[oid] = context

        if oid not in self._finalizers:
            self._finalizers[oid] = weakref.finalize(obj, self._cleanup, oid)

    def __contains__(self, obj: K) -> bool:
        return id(obj) in self._data

    def __getitem__(self, obj: K) -> V:
        oid = id(obj)
        return self._data[oid]

    def get(self, obj: K, default: V | None = None) -> V | None:
        return self._data.get(id(obj), default)

    def __delitem__(self, obj: K) -> None:
        """
        Delete the entry for obj using 'del wc[obj]'.
        """
        oid = id(obj)
        self._data.pop(oid)

        finalizer = self._finalizers.pop(oid, None)
        if finalizer is not None:
            finalizer.detach()

    def _cleanup(self, oid: int) -> None:
        """Called automatically when the object is garbage-collected."""
        self._data.pop(oid, None)
        self._finalizers.pop(oid, None)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"WeakContextDictionary({self._data})"



Elem = TypeVar("Elem")

"""
A processor is a method receives and outputs an Elem of a given type.
If a new Elem is generated, returns it, otherwise if nothing changes, returns None.
Hence to always get the final result, one can use:
    Processor(elem, param) or elem
"""


@dataclasses.dataclass
class ProcessFailureReason:
    reason: str = ""
    trivial: bool = True

Processor = Callable[[Elem], Elem | ProcessFailureReason | None]


class ProcessorCollection(Generic[Elem]):
    """
    Registers processors into a list, and runs them all when trying to process an element.
    There are 3 types or processors:
    1. Generic processors,
    2. Specific element processor: only runs for a specific element (use id(elem) to specify it)
    3. Specific class processor: only runs if the element is from a specific class

    Both (2) and (3) can be implemented using generic processors, but are added here for simplicity of use.
    """

    def __init__(self):
        self.processors: list[Processor] = []
        self.class_processors: dict[type, list[Processor]] = {}
        self.final_version = WeakContextDictionary[Elem, Elem]()
        self._process_stage = WeakContextDictionary[Elem, bool]()

    def reset_element(self, element: Elem) -> None:
        if element in self.final_version:
            del self.final_version[element]
        if element in self._process_stage:
            del self._process_stage[element]

    def set_final_version(self, element: Elem, final_version_element: Elem):
        self.final_version[element] = final_version_element

    def register_processor(self, processor: Processor) -> None:
        self.processors.append(processor)

    # TODO: make sure that the class processor corresponds to the given map_elem_class
    def register_class_processor(self, elem_class: type[Elem], processor: Processor) -> None:
        key = elem_class
        if key not in self.class_processors:
            self.class_processors[key] = []
        self.class_processors[key].append(processor)

    def one_step_process(self, elem: Elem) -> Elem | None:
        """
        Runs all the registered processors, until one of them updates the element, and returns this result.
        If none of them changes the element, returns None.
        """

        for processor in (
            self.processors + self.class_processors.get(type(elem), [])
        ):

            # TODO: Maybe use __qualname__ instead?
            message = f"Processing {processor.__qualname__} ( {red(elem)} )"
            title_start = f"Step: {processor.__qualname__} ( {red(elem)} )"
            logger.log(message, action=TreeAction.GO_DOWN)

            result = processor(elem)

            result = result or ProcessFailureReason("", False)
            if isinstance(result, ProcessFailureReason):
                if result.reason != "" and not result.trivial:
                    logger.log(message=result.reason)
                    # print(f'Simplification failed because of {result.reason}')
                logger.set_context_title(f'{title_start} = {magenta("- - -")}')
                logger.log(message=f"{magenta('- - -')}", action=TreeAction.GO_UP, delete_context=result.trivial)
                continue

            # result is an Elem type
            logger.set_context_title(f"{title_start} => {green(result)} [{cyan(result.__class__.__name__)}]")
            logger.log(message=f"Produced {green(result)}", action=TreeAction.GO_UP)
            return result

        return None

    def full_process(self, elem: Elem) -> Elem | None:
        """
        Runs all the registered processes again and again until none of them changes the resulting element, and
        returns it. Returns None if there wasn't any change.
        """
        if elem in self.final_version:
            elem_final_version = self.final_version[elem]
            return elem_final_version if (elem_final_version is not elem) else None

        # Make sure not to go into simplification loops
        original_elem = elem
        if self._process_stage.get(original_elem, False):
            logger.log(f'{red("!!!")} looped back to processing {red(original_elem)}')
            return None
        self._process_stage[original_elem] = True

        was_processed = False

        title_start = f"Full: [{cyan(elem.__class__.__name__)}] ( {red(elem)} )"
        message = f"Full Processing ( {red(elem)} ) , [{cyan(elem.__class__.__name__)}]"
        logger.log(message=message, action=TreeAction.GO_DOWN, back=Back.LIGHTBLACK_EX)

        # Run simplification steps
        while True:
            if elem in self.final_version:
                result = self.final_version[elem]
                was_processed = (result is not original_elem)
                elem = result
                break
            result = self.one_step_process(elem)
            if result is None:
                break
            elem = result
            was_processed = True

        self._process_stage[original_elem] = False

        if was_processed:
            logger.set_context_title(f"{title_start} => {green(elem)}")
            logger.log(f"Full Produced {green(elem)}", action=TreeAction.GO_UP)
            self.final_version[original_elem] = elem
            self.final_version[elem] = elem
            return elem
        else:
            logger.set_context_title(f'{title_start} = {magenta("X X X")}')
            logger.log(f'{magenta("X X X")} ', action=TreeAction.GO_UP)
            self.final_version[original_elem] = elem
            return None

