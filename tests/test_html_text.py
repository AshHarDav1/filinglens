from filinglens.ingestion.html_text import html_to_text


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
        "Net income $ 5,363",
        "Capex $ 1,577",
    ]


def test_skips_hidden_inline_xbrl_header():
    html = """
    <body><div style="display:none"><ix:header><ix:hidden>dei:EntityCentralIndexKey 0000002488
    </ix:hidden></ix:header></div><p>Visible text</p></body>
    """
    assert html_to_text(html) == "Visible text"
