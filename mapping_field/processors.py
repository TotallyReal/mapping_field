from typing import TypeVar, Generic, Callable, Optional, List, Tuple, Type, Dict

Elem = TypeVar('Elem')
Param = TypeVar('Param')

"""
A processor is a method which processes a given Elem using the parameters in Param.
If a new Elem is generated, returns it, otherwise if nothing changes, returns None.
Hence to always get the final result, one can use:
    Processor(elem, param) or elem
"""
Processor = Callable[[Elem, Param], Optional[Elem]]

ParamProcessor = Callable[[Param], Optional[Elem]]

class ProcessorCollection(Generic[Elem, Param]):
    """
    Registers processors into a list, and runs them all when trying to process an element.
    There are 3 types or processors:
    1. Generic processors,
    2. Specific element processor: only runs for a specific element (use id(elem) to specify it)
    3. Specific class processor: only runs if the element is from a specific class

    Both (2) and (3) can be implemented using generic processors, but are added here for simplicity of use.
    """

    def __init__(self):
        self.processors: List[Processor] = []
        self.elem_processors: Dict[int, List[ParamProcessor]] = {}
        self.class_processors: Dict[type, List[Processor]] = {}

    def register_processor(self, processor: Processor) -> None:
        self.processors.append(processor)

    def register_elem_processor(self, elem: Elem, processor: ParamProcessor) -> None:
        key = id(elem)
        if key not in self.elem_processors:
            self.elem_processors[key] = []
        self.elem_processors[key].append(processor)

    # TODO: make sure that the class processor corresponds to the given map_elem_class
    def register_class_processor(self, elem_class: Type[Elem], processor: Processor) -> None:
        key = elem_class
        if key not in self.elem_processors:
            self.class_processors[key] = []
        self.class_processors[key].append(processor)

    def one_step_process(self, elem: Elem, param: Param) -> Optional[Elem]:
        """
        Runs all the registered processors, until one of them updates the element, and returns this result.
        If none of them changes the element, returns None.
        """
        for processor in self.processors:
            result = processor(elem, param)
            if result is None:
                continue
            return result

        for processor in self.elem_processors.get(id(elem), []):
            result = processor(param)
            if result is None:
                continue
            return result

        for processor in self.class_processors.get(elem.__class__, []):
            result = processor(elem, param)
            if result is None:
                continue
            return result

        return None

    def full_process(self, elem: Elem, param: Param) -> Optional[Elem]:
        """
        Runs all the registered processes again and again until none of them changes the resulting element, and
        returns it. Returns None if there wasn't any change.
        """
        is_simpler = False
        while True:
            result = self.one_step_process(elem, param)
            if result is None:
                break
            elem = result
            is_simpler = True

        return elem if is_simpler else None