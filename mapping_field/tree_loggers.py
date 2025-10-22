from enum import Enum, auto

import logging

class TreeAction(Enum):
    NEUTRAL = auto()
    GO_DOWN = auto()
    GO_UP = auto()

class TreeLogger:
    # TODO: sometime, if I really need it, add a "SharedTreePosition" so I could make loggers for different trees.
    _depth = 0  # shared depth for all instances

    def __init__(self, name="TreeLogger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def log(self, message, action=TreeAction.NEUTRAL):
        initial = '| ' * TreeLogger._depth
        if action == TreeAction.GO_DOWN:
            initial += '┌ '
            TreeLogger._depth += 1
        elif action == TreeAction.GO_UP:
            TreeLogger._depth = max(0, TreeLogger._depth - 1)
            initial = '| ' * TreeLogger._depth + '└ '
        self.logger.info(f'{initial}{message}')
