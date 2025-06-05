import pandas as pd
import textwrap
import math
import itertools

def format_prop(label: str, count: int, total: int) -> str:
    return f"{count} {label} ({100*count/total:.2f}%)"

def show_list(lines, widths, formats=None, indent=1, with_index=False, headers=None):
    if formats is None:
        formats = ('',)*len(widths)
    indent_width = indent * 4
    if with_index:
        lines_iter = ([i+1, *data] for i, data in enumerate(lines))
        index_width = math.floor(1 + math.log10(1 + len(lines)))
        line_widths = (index_width, *widths)
        line_formats = ('', *formats)
    else:
        lines_iter = lines
        line_widths = widths
        line_formats = formats
    line_formats = [
        str(w)+f
        for w, f in zip(line_widths, line_formats)
    ]
    formatted_list = (
        make_line(line, line_formats, indent_width)
        for line in lines_iter
    )
    if headers is None:
        formatted_headers = []
    else:
        full_header = ['', *headers] if with_index else headers
        total_width = indent_width + sum(line_widths) + len(line_widths) - 1
        formatted_headers = [
            make_line(full_header, [str(x) for x in line_widths], indent_width),
            '-'*total_width,
        ]
    return  '\n'.join(itertools.chain(formatted_headers, formatted_list))

def make_line(line, formats, indent_width):
    indent_str = ' '*indent_width
    formatted_items = (
        format(x, f)
        for x, f in zip(line, formats)
    )
    return indent_str+' '.join(formatted_items)

def fix_length(data, size):
    line = str(data).replace('\n', ' ') if not pd.isna(data) else ''
    if len(line) > size:
        return line[:size-1]+'…'
    return line.ljust(size)

def show_posts(posts, indent=1):
    return '\n'.join(' '*(4*indent)+l for l in _show_posts_iter(posts))

def _show_posts_iter(posts):
    for post_data in posts.itertuples():
        has_url = False
        yield "┏" + "━" * 8 + "┯" + "━" * 69 + "┓"
        yield f"┃        │ r/{fix_length(post_data.subreddit, 65)} ┃"
        title_lines = textwrap.wrap(post_data.title, width=67)
        score_index = len(title_lines) // 2
        for i, line in enumerate(title_lines):
            s = post_data.score if i == score_index else ''
            yield f"┃ {s:6} │ {fix_length(line, 67)} ┃"
        yield "┠" + "─" * 8 + "┴" + "─" * 69 + "┨"
        if not pd.isna(post_data.text):
            for line in textwrap.wrap(post_data.text, width=76):
                yield f"┃ {fix_length(line, 76)} ┃"
        yield ("┣" if has_url else "┗") + "━" * 78+ "┛"
        if has_url:
            yield f"┗➤ {post_data.url}"
        yield ''
