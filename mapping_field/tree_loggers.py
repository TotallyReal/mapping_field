from colorama import init, Fore, Back, Style
from enum import Enum, auto

import logging

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

class TreeLogger:
    # TODO: sometime, if I really need it, add a "SharedTreePosition" so I could make loggers for different trees.
    _depth = 0  # shared depth for all instances
    _max_depth = 30

    @classmethod
    def reset(cls):
        TreeLogger._depth = 0

    def __init__(self, name="TreeLogger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def log(self, message, action=TreeAction.NEUTRAL):
        tab_symbols = ['| '] * TreeLogger._depth
        if action == TreeAction.GO_DOWN:
            tab_symbols.append('┌ ')
            TreeLogger._depth += 1
        elif action == TreeAction.GO_UP:
            TreeLogger._depth = max(0, TreeLogger._depth - 1)
            tab_symbols[-1] = '└ '
        tab_symbols = [(f'{Back.LIGHTBLACK_EX}{c}{Style.RESET_ALL}' if i%3==0 else c) for i, c in enumerate(tab_symbols)]
        initial = ''.join(tab_symbols)
        self.logger.info(f'{initial}{message}')
        if TreeLogger._depth > TreeLogger._max_depth:
            raise Exception('Too many recursive logs')
