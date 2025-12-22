from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Static

from mapping_field.global_config import PROJECT_ROOT, file_path_to_module
from mapping_field.log_utils.terminal_launcher import ensure_real_terminal
from mapping_field.log_utils.tree_loggers import TreeContext
from mapping_field.utils.serializable import Serializable

# Launch real terminal if needed
ensure_real_terminal()

# -------- CONFIGURATION --------
LOG_DIR = PROJECT_ROOT / "mapping_field/tests/logs"
LOG_FILE_EXTENSION = "log_context"


# -------- UTILITY FUNCTIONS --------
def get_latest_log_file() -> Path:
    """Return the latest *.log file in LOG_DIR"""
    log_files = sorted(LOG_DIR.glob(f"*.{LOG_FILE_EXTENSION}"), key=lambda f: f.stat().st_mtime, reverse=True)
    return log_files[0] if log_files else None


# -------- APP --------
class LogViewer(App):

    def compose(self) -> ComposeResult:
        yield Static("", id="view")

    def on_mount(self):
        self.load_latest_log()
        self.update_view()

    def update_view(self):
        if self.latest_file is None:
            header = Text(f"Could not find a {LOG_FILE_EXTENSION} file in {LOG_DIR} ")
            lines_rich = ""
        else:
            filename = file_path_to_module(self.latest_file)
            header = Text(
                f"{filename} | [depth = {len(self.context_path)}, log_count={self.tree_context.information_count}]\n",
                style="bold",
            )

            lines_rich = "\n".join(self.collect_lines(0, self.tree_context))

        header.append(lines_rich)

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
        elif event.key == "shift+left":
            if self.position_jumps[-1] < len(self.context_path) - 1:
                self.position_jumps.append(len(self.context_path) - 1)
        elif event.key == "shift+right":
            if len(self.position_jumps) > 1:
                self.position_jumps.pop(-1)
        elif event.key == "left":
            if len(self.context_path) > 1:
                self.current_context = self.current_context.parent
                if self.position_jumps[-1] == len(self.context_path) - 1:
                    self.position_jumps.pop(-1)
                self.context_path.pop()
        elif event.key == "q":
            self.exit()
        elif event.key == "r":  # reload latest log
            self.load_latest_log()
        self.context_path[-1] = max(0, min(len(self.current_context.information) - 1, self.context_path[-1]))
        self.update_view()

    def load_latest_log(self):
        self.latest_file = get_latest_log_file()
        if self.latest_file is not None:
            self.tree_context = Serializable.load_element(self.latest_file)
            self.current_context = self.tree_context
            self.context_path = [0]
            self.position_jumps = [0]
            self.line_index = 0

    def collect_lines(self, path_position: int, context: TreeContext, tab_count: int = 0):
        information = context.information
        cur_context_pos = self.context_path[path_position]

        if path_position < self.position_jumps[-1]:
            lines = self.collect_lines(path_position + 1, information[cur_context_pos], tab_count)
            lines = [f"| {line}" for line in lines]
            return lines

        tabs = "    " * tab_count
        lines = [
            f'{tabs}{"+" if isinstance(single, TreeContext) else ""}{single}' for single in information
        ]

        if path_position == len(self.context_path) - 1:
            # This is the most inner context
            lines[cur_context_pos] = tabs + " > " + str(information[cur_context_pos])
            lines = [""] + lines + [""]
        else:
            lines = (
                    lines[:cur_context_pos] +
                    self.collect_lines(path_position + 1, information[cur_context_pos], tab_count + 1) +
                    lines[cur_context_pos+1:]
            )

        return lines


# -------- MAIN --------
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        LOG_DIR = Path(sys.argv[1])
        assert LOG_DIR.is_dir()
    LogViewer().run()
