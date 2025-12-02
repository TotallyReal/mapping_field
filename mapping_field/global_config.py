from pathlib import Path

# -------------------------
# Core project paths
# -------------------------
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "mapping_field"
PYTHON_EXEC = PROJECT_ROOT / ".venv/bin/python"


# -------------------------
# Utility: convert file path to Python module path
# -------------------------
def file_path_to_module(file_path: Path) -> str:
    """
    Convert a file path inside the project to a module path.
    suitable for `python -m module.path`.

    Example:
        file_path = PROJECT_ROOT / mapping_field/foo/bar.py
        returns "mapping_field.foo.bar"
    """
    file_path = Path(file_path).resolve()
    try:
        relative_path = file_path.relative_to(PROJECT_ROOT)
    except ValueError:
        # fallback if script is outside project root
        return str(file_path)
    return ".".join(relative_path.with_suffix("").parts)
