"""Turn a 10-K HTML document into plain text (stdlib only).

Tables keep one line per row, cells separated by " | ", so a label stays next to its numbers:
"Total cost of revenue | 32,780 | 33,038 | 27,078". Splitting into sections is parse_10k's job.
"""

import re
from html.parser import HTMLParser

_SKIP = {"script", "style", "head", "title", "ix:header"}  # ix:header = hidden inline-XBRL data
_BLOCK = {"p", "div", "br", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6", "section"}
_CELL = {"td", "th"}
# Cells that only carry formatting of a neighbour: "$" before a number, ")" and "%" after it
_DROP_CELLS = {"$"}
_ATTACH_TO_PREVIOUS = {")", "%", ")%"}


def format_row(cells: list[str]) -> str:
    """Join a table row's cells: drop empty spacer and "$" cells, re-attach ")" and "%"."""
    out: list[str] = []
    for raw in cells:
        cell = " ".join(raw.split())
        if not cell or cell in _DROP_CELLS:
            continue
        if cell in _ATTACH_TO_PREVIOUS and out:
            out[-1] += cell
        else:
            out.append(cell)
    return " | ".join(out)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.table_depth = 0
        self.row: list[str] | None = None  # cells of the current table row
        self.cell: list[str] | None = None  # text of the current cell

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP:
            self.skip_depth += 1
        elif self.cell is not None:
            # Inside a cell everything (paragraphs, line breaks, nested tables) is one line
            if tag in _BLOCK or tag in _CELL:
                self.cell.append(" ")
            if tag == "table":
                self.table_depth += 1
        elif tag == "table":
            self.table_depth += 1
            self.parts.append("\n")
        elif tag == "tr" and self.table_depth:
            self.row = []
        elif tag in _CELL and self.row is not None:
            self.cell = []
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag == "table" and self.table_depth > 1:
            self.table_depth -= 1  # a nested table ends; we are still inside the outer cell
        elif tag in _CELL and self.cell is not None:
            self.row.append("".join(self.cell))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.cell is not None:  # unclosed <td>
                self.row.append("".join(self.cell))
                self.cell = None
            line = format_row(self.row)
            if line:
                self.parts.append("\n" + line + "\n")
            self.row = None
        elif tag == "table" and self.table_depth:
            self.table_depth -= 1
            self.parts.append("\n")
        elif tag in _BLOCK and self.cell is None:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.row is not None:
            return  # whitespace between cells: the source of "one value per line" in M0
        else:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
