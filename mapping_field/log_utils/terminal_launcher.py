# terminal_launcher.py
import os
import sys
import subprocess
from shlex import quote

def ensure_real_terminal():
    """If running inside PyCharm or another non-TTY, open a real macOS Terminal."""
    if sys.stdin.isatty():
        return  # Already in a real terminal

    script = os.path.abspath(sys.argv[0])
    python_exec = quote(sys.executable)
    command = f'{python_exec} {quote(script)}'

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
        end tell
        '''
        subprocess.run(["osascript", "-e", applescript])
        sys.exit(0)

    elif os.name == "nt":  # Windows
        subprocess.run(
            ["start", "cmd", "/k", f"{python_exec} {quote(script)}"],
            shell=True
        )
        sys.exit(0)

    else:
        print("No real terminal detected and platform not supported.")
        sys.exit(1)
