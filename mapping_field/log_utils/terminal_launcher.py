import os
import subprocess
import sys

from pathlib import Path

from mapping_field.global_config import PROJECT_ROOT, PYTHON_EXEC, file_path_to_module


def ensure_real_terminal():
    """If running inside PyCharm or another non-TTY, open a real macOS Terminal."""
    if sys.stdin.isatty():
        return  # Already in a real terminal

    script_path = Path(sys.argv[0]).resolve()
    module_name = file_path_to_module(script_path)

    command = f'cd "{PROJECT_ROOT}" && "{PYTHON_EXEC}" -m "{module_name}"'
    command = command.replace('"', '\\"')  # escape any double quotes

    if sys.platform == "darwin":  # macOS
        applescript = f'''
        tell application "Terminal"
            if not (exists window 1) then
                do script "{command}"
            else
                do script "{command}" in window 1
            end if
            set bounds of front window to {100, 100, 1200, 800} 
            activate
            set current settings of front window to settings set "Logger"
            set custom title of front window to "Logger Terminal"
            # do script "tput rmam" in front window
        end tell
        '''
        subprocess.run(["osascript", "-e", applescript])
        sys.exit(0)

    elif os.name == "nt":  # Windows
        # TODO: run this in windows and see if it works
        subprocess.run(
            ["start", "cmd", "/k", f'"{PYTHON_EXEC}" -m "{module_name}"'],
            shell=True
        )
        sys.exit(0)

    else:
        print("No real terminal detected and platform not supported.")
        sys.exit(1)
