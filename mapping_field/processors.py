import dataclasses

from colorama import init, Fore, Style, Back
from typing import TypeVar, Generic, Callable, Optional, List, Type, Dict, Union

from mapping_field.log_utils.tree_loggers import TreeLogger, TreeAction, green, red, yellow, magenta, cyan

init(autoreset=True)
logger = TreeLogger(__name__)

Elem = TypeVar('Elem')
Param = TypeVar('Param')

"""
A processor is a method which processes a given Elem using the parameters in Param.
If a new Elem is generated, returns it, otherwise if nothing changes, returns None.
Hence to always get the final result, one can use:
    Processor(elem, param) or elem
"""
@dataclasses.dataclass
class ProcessFailureReason:
    reason: str = ''
    trivial: bool = True

Processor = Callable[[Elem, Param], Optional[Union[Elem, ProcessFailureReason]]]

ParamProcessor = Callable[[Param], Optional[Union[Elem, ProcessFailureReason]]]

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
        self.elem_processors[key].append(named_forgetful_function(processor))

    # TODO: make sure that the class processor corresponds to the given map_elem_class
    def register_class_processor(self, elem_class: Type[Elem], processor: Processor) -> None:
        key = elem_class
        if key not in self.class_processors:
            self.class_processors[key] = []
        self.class_processors[key].append(processor)

    def one_step_process(self, elem: Elem, param: Param) -> Optional[Elem]:
        """
        Runs all the registered processors, until one of them updates the element, and returns this result.
        If none of them changes the element, returns None.
        """

        for processor in self.processors + self.elem_processors.get(id(elem), []) + self.class_processors.get(type(elem), []):

            # TODO: Maybe use __qualname__ instead?
            message = f'Processing {processor.__qualname__} ( {red(elem)} , {yellow(param)} )'
            title_start = f'Step: {processor.__qualname__} ( {red(elem)} , {yellow(param)} )'
            logger.log(message, action=TreeAction.GO_DOWN)
            result = processor(elem, param)

            if result is None:
                result = ProcessFailureReason('', False)
            if isinstance(result, ProcessFailureReason):
                if result.reason != '':
                    pass
                    # print(f'Simplification failed because of {result.reason}')
                logger.set_context_title(f'{title_start} = {magenta("- - -")}')
                logger.log(message=f'{magenta("- - -")}', action=TreeAction.GO_UP, delete_context=result.trivial)
                continue
            logger.set_context_title(f'{title_start} => {green(result)}')
            logger.log(message=f'Produced {green(result)}', action=TreeAction.GO_UP)
            return result

        return None

    def full_process(self, elem: Elem, param: Param) -> Optional[Elem]:
        """
        Runs all the registered processes again and again until none of them changes the resulting element, and
        returns it. Returns None if there wasn't any change.
        """
        was_processed = False

        title_start = f'Full: [{cyan(elem.__class__.__name__)}] ( {red(elem)} , {yellow(param)} )'
        message = f'Full Processing ( {red(elem)} , {yellow(param)} ) , [{cyan(elem.__class__.__name__)}]'
        logger.log(message=message, action=TreeAction.GO_DOWN, back=Back.LIGHTBLACK_EX)
        while True:
            # TODO:
            #   Should I add a mechanism that prevent running the same process that made the change in the
            #   last loop?
            result = self.one_step_process(elem, param)
            if result is None:
                break
            elem = result
            was_processed = True

        if was_processed:
            logger.set_context_title(f'{title_start} => {green(elem)}')
            logger.log(f'Full Produced {green(elem)}', action=TreeAction.GO_UP)
            return elem
        logger.set_context_title(f'{title_start} = {magenta("X X X")}')
        logger.log(f'{magenta("X X X")} ', action=TreeAction.GO_UP)
        return None

def named_forgetful_function(func: ParamProcessor) -> Processor:
    def wrapper(elem: Elem, param: Param) -> Optional[Union[Elem, ProcessFailureReason]]:
        return func(param)

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    return wrapper
