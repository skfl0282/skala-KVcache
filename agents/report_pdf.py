"""마크다운 보고서를 PDF로 저장하는 공용 유틸리티.

PyMuPDF의 Story(HTML -> PDF)를 쓰므로 별도 시스템 도구(pandoc, LaTeX 등)가
필요 없고, 한글은 PyMuPDF 내장 폴백 폰트로 출력된다.
"""

import pymupdf
from markdown_it import MarkdownIt

MAX_PAGES = 10  # 과제 분량 제한
# 분량을 맞추기 위해 차례로 시도하는 본문 글자 크기(pt). 마지막 값보다 작게는 줄이지 않는다.
FONT_SIZES = (10.5, 10, 9.5, 9)
MARGIN = 50  # 페이지 여백(pt)

_CSS = """
body {{ font-family: sans-serif; font-size: {size}pt; line-height: 1.45; }}
h1 {{ font-size: 1.7em; }}
h2 {{ font-size: 1.35em; margin-top: 1em; }}
h3 {{ font-size: 1.15em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 0.5pt solid #888; padding: 3pt; font-size: 0.9em; }}
th {{ background-color: #eee; }}
"""

_markdown = MarkdownIt("commonmark").enable("table")


def _render(html: str, path: str, size: float) -> int:
    """HTML을 A4 PDF로 쓰고 페이지 수를 반환한다."""
    story = pymupdf.Story(html=html, user_css=_CSS.format(size=size))
    writer = pymupdf.DocumentWriter(path)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (MARGIN, MARGIN, -MARGIN, -MARGIN)
    pages, more = 0, 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()
    return pages


def save_report_pdf(markdown: str, path: str, max_pages: int = MAX_PAGES) -> int:
    """마크다운 보고서를 PDF로 저장하고 페이지 수를 반환한다.

    max_pages 안에 들어올 때까지 글자 크기를 FONT_SIZES 순서로 줄인다. 가장 작은
    크기로도 넘치면 그 크기로 저장하고 경고를 출력한다 (분량 초과 여부는 반환값으로
    호출 측이 판단).
    """
    html = _markdown.render(markdown)
    for size in FONT_SIZES:
        pages = _render(html, path, size)
        if pages <= max_pages:
            break
    else:
        print(f"[WARN] 보고서가 {pages}쪽으로 분량 제한({max_pages}쪽)을 넘습니다: {path}")
    return pages
