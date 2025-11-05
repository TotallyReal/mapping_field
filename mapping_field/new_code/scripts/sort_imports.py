from pathlib import Path

import isort

# Set the root of your project
project_root = Path(__file__).parent.parent

# Recursively sort all Python files
for py_file in project_root.rglob("*.py"):
    isort.file(
        str(py_file),
        multi_line_output=5,
        include_trailing_comma=True,
        force_grid_wrap=0,
        use_parentheses=True,
        line_length=100,
        combine_as_imports=True,
        known_first_party=['mapping_field'],
        lines_between_types=1,
    )

print("Imports sorted successfully!")

