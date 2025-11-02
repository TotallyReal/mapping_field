from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Box:
    hor: str
    vert: str
    tl: str
    tr: str
    bl: str
    br: str


# 4 box styles
SINGLE = Box('─', '│', '┌', '┐', '└', '┘')
DOUBLE = Box('═', '║', '╔', '╗', '╚', '╝')
HEAVY  = Box('━', '┃', '┏', '┓', '┗', '┛')
ROUND  = Box('─', '│', '╭', '╮', '╰', '╯')


def draw_box(text: str, box: Box, width: Optional[int] = None, height: Optional[int] = None) -> str:
    """
    Draw a text box with the given style and optional size constraints.
    """

    lines = text.splitlines() or ['']
    max_text_width = max(len(line) for line in lines)
    text_height = len(lines)

    inner_width = max(width - 2 if width else 0, max_text_width)
    inner_height = max(height - 2 if height else 0, text_height)

    # Center vertically and horizontally
    top = box.tl + box.hor * inner_width + box.tr
    bottom = box.bl + box.hor * inner_width + box.br

    padding_top = (inner_height - text_height) // 2
    padding_bottom = inner_height - text_height - padding_top

    content_lines = [top]
    content_lines.extend([box.vert + ' ' * inner_width + box.vert] * padding_top)

    for line in lines:
        pad_left = (inner_width - len(line)) // 2
        pad_right = inner_width - len(line) - pad_left
        content_lines.append(box.vert + ' ' * pad_left + line + ' ' * pad_right + box.vert)

    content_lines.extend([box.vert + ' ' * inner_width + box.vert] * padding_bottom)
    content_lines.append(bottom)

    return '\n'.join(content_lines)


# --- Example usage ---
if __name__ == "__main__":
    print(draw_box("Hello!", SINGLE, width=12))
    print()
    print(draw_box("Important Message\nI am a box", DOUBLE, width=30, height=6))
    print()
    print(draw_box("Heavy Border", HEAVY, height=12))
    print()
    print(draw_box("Rounded Corners", ROUND))
