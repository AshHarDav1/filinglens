from filinglens.ingestion.html_text import format_row, html_to_text


def test_extracts_paragraphs_and_table_rows():
    html = """
    <html><head><title>ignored</title><style>p {color: red}</style></head>
    <body>
      <p>Item 7.&nbsp;Management's Discussion</p>
      <div>Revenue   increased <b>12%</b>.</div>
      <table>
        <tr><td>Net income</td><td>$</td><td>5,363</td></tr>
        <tr><td>Capex</td><td>$</td><td>1,577</td></tr>
      </table>
    </body></html>
    """
    lines = [line for line in html_to_text(html).splitlines() if line]
    assert lines == [
        "Item 7. Management's Discussion",
        "Revenue increased 12%.",
        "Net income | 5,363",
        "Capex | 1,577",
    ]


def test_real_microsoft_row_stays_on_one_line():
    # Shortened from Microsoft's FY2016 10-K: a <P> in the label cell, spacer cells, and a line
    # break between every cell, which M0 turned into one value per line.
    html = """<table><TR BGCOLOR="#cceeff">
<TD VALIGN="top"> <P STYLE="margin-left:3.00em"><FONT SIZE="2">Total cost of revenue</FONT></P></TD>
<TD VALIGN="bottom"><FONT SIZE="1">&nbsp;&nbsp;</FONT></TD>
<TD VALIGN="bottom"><FONT SIZE="2"><B>&nbsp;</B></FONT></TD>
<TD VALIGN="bottom" ALIGN="right"><FONT SIZE="2"><B>32,780</B></FONT></TD>
<TD NOWRAP VALIGN="bottom"><FONT SIZE="2"><B>&nbsp;&nbsp;</B></FONT></TD>
<TD VALIGN="bottom" ALIGN="right"><FONT SIZE="2">33,038</FONT></TD>
<TD VALIGN="bottom" ALIGN="right"><FONT SIZE="2">27,078</FONT></TD>
</TR></table>"""
    assert html_to_text(html) == "Total cost of revenue | 32,780 | 33,038 | 27,078"


def test_negative_numbers_and_percents_are_reattached():
    assert format_row(["Total", "$", "(431", ")", "", "$", "346"]) == "Total | (431) | 346"
    assert format_row(["Gross margin", "61.6", "%", "64.7", "%"]) == "Gross margin | 61.6% | 64.7%"


def test_block_tags_inside_a_cell_do_not_break_the_row():
    html = "<table><tr><td><div>Cost of</div><div>revenue</div></td><td>1<br>2</td></tr></table>"
    assert html_to_text(html) == "Cost of revenue | 1 2"


def test_skips_hidden_inline_xbrl_header():
    html = """
    <body><div style="display:none"><ix:header><ix:hidden>dei:EntityCentralIndexKey 0000002488
    </ix:hidden></ix:header></div><p>Visible text</p></body>
    """
    assert html_to_text(html) == "Visible text"
