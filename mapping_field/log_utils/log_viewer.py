import math
import os
from pathlib import Path
from typing import List

from colorama import Fore, Style
from textual.app import App, ComposeResult
from textual.widgets import Static
from textual import events

from mapping_field.log_utils.tree_loggers import TreeContext
from mapping_field.serializable import Serializable
from mapping_field.log_utils.terminal_launcher import ensure_real_terminal
from rich.text import Text
from rich.console import Console
from rich.ansi import AnsiDecoder

# Launch real terminal if needed
ensure_real_terminal()

decoder = AnsiDecoder()
console = Console()

def ansi_to_rich(text: str) -> Text:
    return Text.assemble(*decoder.decode(text))

# -------- CONFIGURATION --------
LOG_DIR = Path(__file__).parent.parent /'new_code'/'tests'/'logs'
LOG_FILE_EXTENSION = 'log_context'

# -------- UTILITY FUNCTIONS --------
def get_latest_log_file() -> Path:
    """Return the latest *.log file in LOG_DIR"""
    log_files = sorted(LOG_DIR.glob(f'*.{LOG_FILE_EXTENSION}'), key=lambda f: f.stat().st_mtime, reverse=True)
    return log_files[0] if log_files else None

def read_log_file(file_path: Path) -> List[str]:
    """Return list of lines from a log file"""
    with open(file_path, "r") as f:
        return [line.rstrip("\n") for line in f]

def show_lines(log_lines: List[str], line_number: int) -> List[str]:
    """Return the lines to display around line_number"""
    start = max(0, line_number - 2)
    end = min(len(log_lines), line_number + 3)
    return log_lines[start:end]

# -------- APP --------
class LogViewer(App):

    def compose(self) -> ComposeResult:
        yield Static("", id="view")

    def on_mount(self):
        self.load_latest_log()
        self.update_view()

    def update_view(self):

        # build the colored lines (Text)
        lines_rich = ("\n".join(self.collect_lines(0, self.tree_context, tab_count=0)))

        # build a header Text (bold file name + line count)
        filename = getattr(self, "latest_file", "No log")
        from mapping_field.global_config import file_path_to_module
        if filename != 'No log':
            filename = file_path_to_module(filename)
        header = Text(f"{filename} | [depth = {len(self.context_path)}]\n", style="bold")

        # append the ANSI->Rich converted body
        header.append(lines_rich)

        # update the widget with a single Text object (preserves styles)
        self.query_one("#view", Static).update(header)

    def on_key(self, event: events.Key):
        if event.key == "up":
            self.context_path[-1] -= 1
        elif event.key == "down":
            self.context_path[-1] += 1
        elif event.key == "right":
            if isinstance(self.current_context.information[self.context_path[-1]], TreeContext):
                self.current_context = self.current_context.information[self.context_path[-1]]
                self.context_path.append(0)
            else:
                return
        elif event.key == "left":
            if len(self.context_path) > 1:
                self.current_context = self.current_context.parent
                self.context_path.pop()
        elif event.key == "q":
            self.exit()
        elif event.key == "r":  # reload latest log
            self.load_latest_log()
        self.context_path[-1] = max(0, min(len(self.current_context.information)-1, self.context_path[-1]))
        self.update_view()

    def load_latest_log(self):
        latest_file = get_latest_log_file()
        if latest_file is None:
            self.log_lines = ["No log files found."]
        else:
            self.tree_context = Serializable.load_element(latest_file)
            self.current_context = self.tree_context
            self.context_path = [0]
            self.log_lines = read_log_file(latest_file)
            self.line_index = 0
            self.latest_file = latest_file

    def collect_lines(self, path_position:int, context:TreeContext, tab_count: int):
        information = context.information
        tabs = '    '*tab_count
        lines = [tabs + str(single) for single in information[:self.context_path[path_position]]]
        if path_position < len(self.context_path)-1:
            lines += self.collect_lines(path_position+1, information[self.context_path[path_position]], tab_count + 1)
        else:
            lines.append(tabs + ' > ' + str(information[self.context_path[path_position]]))
        lines += [tabs + str(single) for single in information[self.context_path[path_position]+1:]]
        if path_position == len(self.context_path)-1:
            lines = [''] + lines + ['']
        return lines

# -------- MAIN --------
if __name__ == "__main__":
    LogViewer().run()

