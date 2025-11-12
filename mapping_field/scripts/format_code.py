import black
import isort

from mapping_field.global_config import PROJECT_ROOT

src_root = PROJECT_ROOT / "mapping_field"
ignore_dir = src_root / "old_code"

# print("\n ================= Running black ================= ")
# for py_file in src_root.rglob("*.py"):
#     if py_file.is_relative_to(ignore_dir):
#         continue
#     black.format_file_in_place(
#         py_file,
#         fast=False,
#         mode=black.FileMode(line_length=120),
#         write_back=black.WriteBack.YES,
#     )

print("\n ================= Running isort ================= ")
for py_file in src_root.rglob("*.py"):
    if py_file.is_relative_to(ignore_dir):
        continue
    isort.file(
        str(py_file),
        multi_line_output=5,
        include_trailing_comma=True,
        force_grid_wrap=0,
        use_parentheses=True,
        line_length=100,
        combine_as_imports=True,
        known_first_party=["mapping_field"],
        lines_between_types=1,
    )

print("\n ================= Finished code format ================= ")
