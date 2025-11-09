from typing import Union, List, Optional

from colorama import init, Fore, Back, Style
from enum import Enum, auto

import logging

from mapping_field.serializable import DefaultSerializable


def red(x) -> str:
    return f'{Fore.RED}{x}{Style.RESET_ALL}'

def green(x) -> str:
    return f'{Fore.GREEN}{x}{Style.RESET_ALL}'

def blue(x) -> str:
    return f'{Fore.BLUE}{x}{Style.RESET_ALL}'

def yellow(x) -> str:
    return f'{Fore.YELLOW}{x}{Style.RESET_ALL}'

def magenta(x) -> str:
    return f'{Fore.MAGENTA}{x}{Style.RESET_ALL}'

def cyan(x) -> str:
    return f'{Fore.CYAN}{x}{Style.RESET_ALL}'

class TreeAction(Enum):
    NEUTRAL = auto()
    GO_DOWN = auto()
    GO_UP = auto()

Information = Union['TreeContext', str]


class TreeContext(DefaultSerializable):
    def __init__(self, information: Optional[List[Information]] = None, title: str = ''):
        self.parent : Optional['TreeContext'] = None
        self.information = [] if information is None else information
        for context in self.information:
            if isinstance(context, TreeContext):
                context.parent = self
        self.title = title

    def add_information(self, information: Information):
        self.information.append(information)
        if isinstance(information, TreeContext):
            information.parent = self

    def __repr__(self):
        if len(self.title)>0:
            return self.title
        return 'empty' if len(self.information) == 0 else str(self.information[0])

    def set_title(self, title: str):
        self.title = title

class LogTree:

    def __init__(self, name: str):
        self.depth = 0
        self.max_depth = 60
        self.paused = False
        self.log_count = 0
        self.max_log_count = -1
        self.tab_styles = []
        self.context = TreeContext()
        self.context.add_information(name)

    def open_context(self) -> TreeContext:
        parent = self.context
        self.context = TreeContext()
        parent.add_information(self.context)
        self.depth += 1
        return self.context

    def close_context(self) -> TreeContext:
        self.context = self.context.parent
        assert self.context is not None
        self.depth -= 1
        return self.context

    def reset(self):
        self.depth = 0

simplify_tree = LogTree('Simplify Tree')

class TreeLogger:

    @classmethod
    def set_log_state(cls, paused: bool = False) -> None:
        cls._paused = paused

    def reset(self):
        self.tree.depth = 0

    def __init__(self, name="TreeLogger", log_tree: LogTree = simplify_tree):
        self.tree = log_tree
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def log(self, message, action=TreeAction.NEUTRAL, fore: str = '', back: str = ''):
        if self.tree.paused:
            return
        self.tree.log_count += 1
        if self.tree.log_count >= self.tree.max_log_count > 0:
            raise Exception('Too many logs')
        tab_symbols = ['|'] * self.tree.depth
        if action == TreeAction.GO_DOWN:
            self.tree.open_context().add_information(message)
            self.tree.tab_styles.append(f'{fore}{back}')
            tab_symbols.append('┌>')
        elif action == TreeAction.GO_UP:
            tab_symbols[-1] = '└>'
            self.tree.context.add_information(message)
            self.tree.close_context()
        else:
            self.tree.context.add_information(message)
        tab_symbols = [f'{s}{c}{Style.RESET_ALL} ' for c, s in zip(tab_symbols, self.tree.tab_styles)]
        initial = ''.join(tab_symbols)
        self.logger.info(f'{initial}{message}')
        if self.tree.depth > self.tree.max_depth:
            raise Exception('Too many recursive logs')
        if action == TreeAction.GO_UP:
            self.tree.tab_styles.pop(-1)

    def set_context_title(self, title: str):
        self.tree.context.set_title(title)
