"""A fetched page keeps its shape: headings, list items and data tables come out as
Markdown, everything else one line per piece of text as before.

Pinned here:

* a heading is one line with its level ("## Our members"), inline parts joined;
* a list item starts with "- " or its number (an <ol start> counts from there),
  nested items are indented, a <br> inside an item starts a plain next line;
* a data table is one "| a | b |" line per row: a header row gets the "---" line,
  a rowspan repeats its cell down, a colspan leaves blanks, a <br> in a cell is
  " · ", a "|" in a cell is escaped, links stay "label (url)";
* a layout table (nested table, long cells, role=presentation, one row or one
  column) is read as plain text, as before;
* plain paragraphs keep one line per piece of text (address finders need it);
* a very deep page doesn't blow the stack.
"""

from __future__ import annotations

from captain_claw.tools.web_fetch import _extract_readable_text


def _page(body: str, title: str = "Page") -> str:
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


def test_headings_carry_their_level_on_one_line():
    text = _extract_readable_text(_page("<h1>Firms</h1><h3>Our <em>members</em> list</h3>"))
    assert "# Firms" in text.splitlines()
    assert "### Our members list" in text.splitlines()


def test_list_items_are_marked_and_nested_ones_indented():
    text = _extract_readable_text(_page(
        "<ul><li><a href='/'>Home</a></li>"
        "<li><a href='/about'>About</a><ul><li><a href='/team'>Team</a></li></ul></li></ul>"
    ), "https://dir.example/")
    lines = text.splitlines()
    assert "- Home (https://dir.example/)" in lines
    assert "- About (https://dir.example/about)" in lines
    assert "  - Team (https://dir.example/team)" in lines


def test_ordered_list_counts_from_its_start_and_br_starts_a_plain_line():
    text = _extract_readable_text(_page(
        "<ol start='3'><li>Third</li><li><b>Mon–Fri:</b> 9–17<br>Sat 9–13</li></ol>"))
    lines = text.splitlines()
    assert "3. Third" in lines
    assert "4. Mon–Fri: 9–17" in lines
    assert "Sat 9–13" in lines


def test_data_table_rows_keep_their_cells_together():
    text = _extract_readable_text(_page(
        "<table><caption>Members</caption>"
        "<thead><tr><th>Name</th><th>City</th><th>Staff</th></tr></thead>"
        "<tr><td><a href='https://a.example'>Alpha</a></td><td rowspan='2'>Zagreb</td><td>12</td></tr>"
        "<tr><td>Beta</td><td>40</td></tr>"
        "<tr><td>Gamma | Co</td><td>Ilica 12<br>10000 Zagreb</td><td></td></tr>"
        "<tr><td colspan='2'>Total</td><td>52</td></tr>"
        "</table>"))
    lines = text.splitlines()
    at = lines.index("| Name | City | Staff |")
    assert lines[at - 1] == "Members"
    assert lines[at + 1] == "| --- | --- | --- |"
    assert lines[at + 2] == "| Alpha (https://a.example) | Zagreb | 12 |"
    assert lines[at + 3] == "| Beta | Zagreb | 40 |"
    assert lines[at + 4] == "| Gamma \\| Co | Ilica 12 · 10000 Zagreb |  |"
    assert lines[at + 5] == "| Total |  | 52 |"


def test_table_without_a_header_row_has_no_rule_line():
    text = _extract_readable_text(_page(
        "<table><tr><td>Phone</td><td>+385 1 234 567</td></tr>"
        "<tr><td>Email</td><td>info@a.example</td></tr></table>"))
    assert "| Phone | +385 1 234 567 |" in text
    assert "| Email | info@a.example |" in text
    assert "---" not in text


def test_layout_tables_read_as_plain_text():
    long_cell = "<table><tr><td>" + "word " * 300 + "</td><td>side</td></tr><tr><td>a</td><td>b</td></tr></table>"
    nested = "<table><tr><td><table><tr><td>x</td><td>y</td></tr><tr><td>1</td><td>2</td></tr></table></td><td>z</td></tr></table>"
    shown = "<table role='presentation'><tr><td>left</td><td>right</td></tr><tr><td>c</td><td>d</td></tr></table>"
    one_row = "<table><tr><td>logo</td><td>menu</td></tr></table>"
    one_col = "<table><tr><td>first</td></tr><tr><td>second</td></tr></table>"
    for html, plain in ((long_cell, "side"), (shown, "left"), (one_row, "logo"), (one_col, "second")):
        text = _extract_readable_text(_page(html))
        assert "|" not in text, html
        assert plain in text.splitlines()
    # the outer table is layout; the data table inside it still gets its rows
    text = _extract_readable_text(_page(nested))
    assert "| x | y |" in text and "| 1 | 2 |" in text
    assert "z" in text.splitlines()


def test_paragraphs_keep_one_line_per_piece_of_text():
    text = _extract_readable_text(_page("<p>Visit us at <b>Ilica 12, 10000 Zagreb</b> any day</p>"))
    assert "Ilica 12, 10000 Zagreb" in text.splitlines()


def test_title_is_not_repeated_over_a_matching_heading():
    html = "<html><body><h1>Hello</h1><title>Hello</title><p>Body text.</p></body></html>"
    text = _extract_readable_text(html)
    assert text.splitlines()[0] == "# Hello"
    html = "<html><body><h1>Welcome</h1><title>Shop</title><p>x</p></body></html>"
    assert _extract_readable_text(html).splitlines()[:2] == ["Shop", ""]


def test_a_very_deep_page_still_reads():
    html = _page("<div>" * 3000 + "<ul><li>deep item</li></ul>" + "</div>" * 3000)
    assert "- deep item" in _extract_readable_text(html).splitlines()
