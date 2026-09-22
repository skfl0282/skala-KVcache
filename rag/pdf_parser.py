"""알고리즘 기반 학술 논문 PDF 파서 모듈 (DLA + 공간 마스킹 + 다단 정렬 + 수식 정규화)

Vision 딥러닝 모델 없이 PyMuPDF와 pdfplumber의 기하학적 분석만을 이용하여:
1. Figure/Table 영역을 캡션과 함께 250 DPI 고해상도 시각 카드로 추출
2. 도표 내부 라벨 및 파편 텍스트 공간 마스킹 (Spatial Masking)
3. 2단 컬럼 학술 논문의 올바른 읽기 순서(Reading Order) 재정렬
4. LaTeX 수식 블록($$...$$) 및 헤딩(##, ###) 정규화
5. 단락/페이지 경계 하이픈(Dehyphenation) 자동 결합
6. LangChain Document 객체로 변환하여 RAG 파이프라인에 제공
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
import pymupdf
from langchain_core.documents import Document


def safe_rect_union(*rects) -> Optional[pymupdf.Rect]:
    """두께 0의 선분(horizontal/vertical lines)이 포함되어도 안전하게 전체 기하학 영역을 감싸는 유니온 함수."""
    valid = [pymupdf.Rect(r) for r in rects if r is not None and not r.is_infinite]
    if not valid:
        return None
    return pymupdf.Rect(
        min(r.x0 for r in valid),
        min(r.y0 for r in valid),
        max(r.x1 for r in valid),
        max(r.y1 for r in valid),
    )


def dehyphenate(text: str) -> str:
    """줄바꿈이나 단락 경계로 끊어진 하이픈 단어를 복원합니다 (e.g. 'character-\\n\\nistics' -> 'characteristics')."""
    text = re.sub(r"(\b[a-zA-Z]{2,})-\s*\n+\s*([a-zA-Z]{2,}\b)", r"\1\2", text)
    text = re.sub(r"(\b[a-zA-Z]{2,})-\s+([a-z]{2,}\b)", r"\1\2", text)
    return text


def format_bullets_and_lists(text: str) -> str:
    """뭉쳐진 불릿 기호(•)를 마크다운 목록 형식(- )으로 변환합니다."""
    lines = []
    for line in text.splitlines():
        if "•" in line:
            parts = [p.strip() for p in line.split("•") if p.strip()]
            for p in parts:
                lines.append(f"- {p}")
        else:
            lines.append(line)
    return "\n".join(lines)


def detect_heading(text: str) -> Optional[Tuple[int, str]]:
    """'2 Background and Motivation' 또는 '2.1 Architecture' 등의 학술 논문 헤딩을 감지합니다."""
    stripped = text.strip()
    m1 = re.match(r"^(\d+)\.?\s+([A-Z][A-Za-z0-9\s,\-_:]{3,60})$", stripped)
    if m1 and len(stripped.splitlines()) <= 2:
        return 2, f"## {stripped}"

    m2 = re.match(r"^(\d+\.\d+)\.?\s+([A-Z][A-Za-z0-9\s,\-_:]{3,60})$", stripped)
    if m2 and len(stripped.splitlines()) <= 2:
        return 3, f"### {stripped}"

    m3 = re.match(r"^(\d+\.\d+\.\d+)\.?\s+([A-Z][A-Za-z0-9\s,\-_:]{3,60})$", stripped)
    if m3 and len(stripped.splitlines()) <= 2:
        return 4, f"#### {stripped}"

    return None


def parse_equation_block(block_dict: Dict) -> Optional[str]:
    """번호가 매겨진 수식 블록을 감지하여 KaTeX 호환 LaTeX 블록으로 변환합니다:
    e.g. $$ \\mathbf{q}_t = \\mathbf{W}_Q \\mathbf{h}_t \\tag{1} $$
    """
    spans = [s for l in block_dict.get("lines", []) for s in l.get("spans", [])]
    if not spans:
        return None

    all_text = "".join([s["text"] for s in spans]).strip()
    m = re.search(r"\((\d+)\)\s*$", all_text)
    if not m or len(all_text.splitlines()) > 3:
        return None

    # 본문 인용("in Eq. (4)", "Algorithm (1)" 등) 오인식 방지 가드
    if re.search(r"\b(eq\.|equation|algorithm|section|figure|table|in|see|ref|case|step)\s*\((\d+)\)\s*$", all_text, re.IGNORECASE):
        return None

    # 수학 연산자, 관계 기호, 첨자 유무 검증
    math_symbols = set("=+-*/<>≤≥≈≡∈→←∑∏∫√±")
    has_operator = any(c in math_symbols for c in all_text)
    has_subsup = bool(re.search(r"[a-zA-Z_]\s*[\^_{}]", all_text))
    if not has_operator and not has_subsup:
        return None

    tag_num = m.group(1)
    sizes = [s["size"] for s in spans if not re.match(r"^\(\d+\)$", s["text"].strip())]
    if not sizes:
        return None
    base_size = max(sizes)
    ref_y = spans[0]["bbox"][1]

    tokens = []
    for s in spans:
        txt = s["text"].strip()
        if re.match(r"^\(\d+\)$", txt):
            continue
        norm = unicodedata.normalize("NFKD", txt).strip()
        if not norm:
            continue

        is_subscript = s["size"] < base_size * 0.85 and (s["bbox"][1] > ref_y + 1.2)
        is_superscript = s["size"] < base_size * 0.85 and (s["bbox"][1] < ref_y - 1.2)
        is_bold = "bold" in s["font"].lower()

        if is_subscript:
            tokens.append(f"_{{{norm}}}")
        elif is_superscript:
            tokens.append(f"^{{{norm}}}")
        elif is_bold and norm.isalpha() and len(norm) <= 2:
            tokens.append(f"\\mathbf{{{norm}}}")
        else:
            tokens.append(norm)

    formula = " ".join(tokens)
    formula = re.sub(r"\s+(_\{[^\}]+\})", r"\1", formula)
    formula = re.sub(r"\s+(\^\{[^\}]+\})", r"\1", formula)
    formula = re.sub(r"\s*=\s*", " = ", formula)
    formula = formula.rstrip(",").strip()

    return f"$$\n{formula} \\tag{{{tag_num}}}\n$$"


def extract_visuals_and_mask(
    page: pymupdf.Page,
    page_num: int,
    pdf_path: str,
    output_dir: Optional[Path] = None,
) -> Tuple[List[Dict], List[pymupdf.Rect]]:
    """페이지 내 Figure 및 Table을 캡션과 함께 감지하고, 고해상도 시각 카드를 저장하며 마스킹 박스를 반환합니다."""
    page_width = page.rect.width
    page_height = page.rect.height
    blocks = page.get_text("blocks")
    drawings = page.get_drawings()
    words = page.get_text("words")
    raster_images = page.get_image_info()

    detected_visuals = []
    masking_boxes = []

    # 캡션 구분자(: . | —) 필수 정규식
    caption_pattern = re.compile(r"^(Figure|Fig\.|Table)\s+(\d+)\s*[:\.\—\|]", re.IGNORECASE)
    # 본문 인용 서술어 필터
    in_text_verb_pattern = re.compile(
        r"^(Figure|Fig\.|Table)\s+\d+\s+(illustrates|shows|depicts|presents|summarizes|demonstrates|describes|is|are|was|were)\b",
        re.IGNORECASE,
    )

    # pdfplumber 표 추출
    plumber_tables = []
    try:
        with pdfplumber.open(pdf_path) as plumber_pdf:
            if page_num - 1 < len(plumber_pdf.pages):
                plumber_page = plumber_pdf.pages[page_num - 1]
                found_tbls = plumber_page.find_tables()
                for ft in found_tbls:
                    plumber_tables.append({
                        "bbox": pymupdf.Rect(*ft.bbox),
                        "data": ft.extract(),
                    })
    except Exception:
        pass

    for b in blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        stripped_text = text.strip()

        if in_text_verb_pattern.search(stripped_text):
            continue

        match = caption_pattern.search(stripped_text)
        if not match:
            continue

        is_table = "table" in match.group(1).lower()
        kind = "Table" if is_table else "Figure"
        num = match.group(2)
        caption_rect = pymupdf.Rect(x0, y0, x1, y1)

        is_full_width = (caption_rect.width > page_width * 0.6) or (x0 < page_width * 0.25 and x1 > page_width * 0.75)
        is_left_col = (x1 <= page_width * 0.55) and not is_full_width

        if is_full_width:
            min_x, max_x = page_width * 0.05, page_width * 0.95
        elif is_left_col:
            min_x, max_x = page_width * 0.05, page_width * 0.52
        else:
            min_x, max_x = page_width * 0.48, page_width * 0.95

        crop_rect = None
        table_md = None
        candidate_visual_boxes = []

        if is_table:
            # Table: 캡션 아래쪽에 도표 위치
            matching_ptbl = None
            min_dist = float("inf")
            for pt in plumber_tables:
                tb = pt["bbox"]
                if tb.y0 >= caption_rect.y0 - 5 and tb.y0 <= caption_rect.y1 + 100:
                    if tb.intersects(pymupdf.Rect(min_x, 0, max_x, page_height)):
                        dist = abs(tb.y0 - caption_rect.y1)
                        if dist < min_dist:
                            min_dist = dist
                            matching_ptbl = pt

            if matching_ptbl:
                crop_rect = (matching_ptbl["bbox"] + (-4, -4, 4, 4)) & page.rect
                candidate_visual_boxes.append(matching_ptbl["bbox"])
                t_data = matching_ptbl["data"]
                if t_data and len(t_data) > 0:
                    md_rows = []
                    for r_idx, row in enumerate(t_data):
                        cleaned = [str(c).replace("\n", " ").strip() if c is not None else "" for c in row]
                        md_rows.append("| " + " | ".join(cleaned) + " |")
                        if r_idx == 0:
                            md_rows.append("| " + " | ".join(["---"] * len(row)) + " |")
                    table_md = "\n".join(md_rows)
            else:
                candidate_drawings = [
                    d["rect"] for d in drawings
                    if d["rect"].y0 >= caption_rect.y1 - 5
                    and d["rect"].y1 <= caption_rect.y1 + 300
                    and d["rect"].x0 >= min_x - 15
                    and d["rect"].x1 <= max_x + 15
                ]
                if candidate_drawings:
                    candidate_visual_boxes.extend(candidate_drawings)
                    u = safe_rect_union(*candidate_drawings)
                    if u:
                        crop_rect = (u + (-4, -4, 4, 4)) & page.rect
        else:
            # Figure: 캡션 위쪽에 다이어그램 위치
            for d in drawings:
                dr = d["rect"]
                if dr.y1 <= caption_rect.y0 + 10 and dr.y0 >= page_height * 0.05:
                    if dr.x0 >= min_x - 15 and dr.x1 <= max_x + 15:
                        candidate_visual_boxes.append(dr)

            for im in raster_images:
                ir = pymupdf.Rect(im["bbox"])
                if ir.y1 <= caption_rect.y0 + 10 and ir.y0 >= page_height * 0.05:
                    if ir.x0 >= min_x - 15 and ir.x1 <= max_x + 15:
                        candidate_visual_boxes.append(ir)

            if candidate_visual_boxes:
                u = safe_rect_union(*candidate_visual_boxes)
                if u:
                    search_env = u + (-15, -10, 15, 10)
                    words_in_box = []
                    for w in words:
                        wr = pymupdf.Rect(w[:4])
                        if wr.intersects(search_env) and wr.y1 < caption_rect.y0 - 6:
                            if wr.x0 >= min_x - 10 and wr.x1 <= max_x + 10:
                                words_in_box.append(wr)
                    if words_in_box:
                        u = safe_rect_union(u, *words_in_box)

                    crop_rect = (u + (-5, -5, 5, 5)) & page.rect

        if crop_rect is None or crop_rect.width < 20 or crop_rect.height < 20:
            if not candidate_visual_boxes and not is_table:
                continue
            if is_table:
                crop_rect = pymupdf.Rect(min_x, caption_rect.y1 + 5, max_x, caption_rect.y1 + 180) & page.rect
            else:
                candidates_above = [other[3] for other in blocks if other[3] < y0 and (other[3] < y0 - 10)]
                top_bound = max(candidates_above) if candidates_above else (page_height * 0.08)
                crop_rect = pymupdf.Rect(min_x, top_bound - 5, max_x, y0 - 5) & page.rect

        # 캡션과 시각 요소를 결합한 시각 카드 영역 계산
        full_visual_card_rect = safe_rect_union(crop_rect, caption_rect) + (-8, -8, 8, 8)
        full_visual_card_rect = full_visual_card_rect & page.rect

        img_filename = f"{kind.lower()}_{num}.png"
        saved_img_rel_path = None

        if output_dir:
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            img_path = images_dir / img_filename
            try:
                pix = page.get_pixmap(clip=full_visual_card_rect, dpi=250)
                pix.save(str(img_path))
                saved_img_rel_path = str(img_path.relative_to(output_dir.parent))
            except Exception:
                pass

        masking_region = full_visual_card_rect + (-2, -2, 2, 2)
        masking_boxes.append(masking_region)

        detected_visuals.append({
            "kind": kind,
            "num": num,
            "caption": text.strip(),
            "caption_rect": [caption_rect.x0, caption_rect.y0, caption_rect.x1, caption_rect.y1],
            "crop_rect": [full_visual_card_rect.x0, full_visual_card_rect.y0, full_visual_card_rect.x1, full_visual_card_rect.y1],
            "file_name": img_filename,
            "rel_path": saved_img_rel_path or f"images/{img_filename}",
            "table_md": table_md,
        })

    return detected_visuals, masking_boxes


def detect_page_equations(page: pymupdf.Page) -> List[Tuple[pymupdf.Rect, str]]:
    """페이지 내 번호가 매겨진 수식 블록들을 감지하고 수평/수직 조각들을 온전한 KaTeX 수식으로 병합합니다."""
    dict_page = page.get_text("dict")
    text_b = [b for b in dict_page.get("blocks", []) if b.get("type") == 0]
    if not text_b:
        return []

    text_b.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    used_indices = set()
    equations = []

    for idx, b in enumerate(text_b):
        if idx in used_indices:
            continue
        txt = "".join([s["text"] for l in b["lines"] for s in l["spans"]]).strip()
        m = re.search(r"\((\d+)\)\s*$", txt)
        if not m:
            continue
        # 본문 인용 서술어 오인식 방지
        if re.search(
            r"\b(eq\.|equation|algorithm|section|figure|table|in|see|ref|case|step|device|expert|token)\s*\((\d+)\)\s*$",
            txt,
            re.I,
        ):
            continue

        tag_y0, tag_y1 = b["bbox"][1], b["bbox"][3]
        tag_x1 = b["bbox"][2]

        # 1. 동일 수평 라인상의 블록 탐색 (|y0 - tag_y0| < 6)
        line_indices = [
            i
            for i, ob in enumerate(text_b)
            if i not in used_indices
            and abs(ob["bbox"][1] - tag_y0) < 6
            and abs(ob["bbox"][3] - tag_y1) < 6
            and ob["bbox"][0] <= tag_x1 + 5
        ]

        # 우선 동일 라인 블록들로 파싱 시도
        cand_blocks = [text_b[i] for i in line_indices]
        cand_blocks.sort(key=lambda x: x["bbox"][0])
        merged_spans = [s for cb in cand_blocks for l in cb["lines"] for s in l["spans"]]
        merged_rect = pymupdf.Rect(
            min(cb["bbox"][0] for cb in cand_blocks),
            min(cb["bbox"][1] for cb in cand_blocks),
            max(cb["bbox"][2] for cb in cand_blocks),
            max(cb["bbox"][3] for cb in cand_blocks),
        )
        merged_block = {"bbox": tuple(merged_rect), "lines": [{"spans": merged_spans}]}
        res = parse_equation_block(merged_block)

        # 수평 라인만으로 연산자가 부족하거나 분절된 경우에 한해 수직 확장 시도
        if not res and len(line_indices) == 1:
            expanded_indices = list(line_indices)
            for i, ob in enumerate(text_b):
                if i in used_indices or i in line_indices:
                    continue
                ob_txt = "".join([s["text"] for l in ob["lines"] for s in l["spans"]]).strip()
                if len(ob_txt) > 80 or len(ob["lines"]) > 3:
                    continue
                if ob["bbox"][3] >= tag_y0 - 15 and ob["bbox"][1] <= tag_y1 + 15:
                    if ob["bbox"][2] <= tag_x1 + 5 and not re.search(r"\(\d+\)\s*$", ob_txt):
                        expanded_indices.append(i)

            if len(expanded_indices) > 1:
                cand_blocks = [text_b[i] for i in expanded_indices]
                cand_blocks.sort(key=lambda x: (x["bbox"][0], x["bbox"][1]))
                merged_spans = [s for cb in cand_blocks for l in cb["lines"] for s in l["spans"]]
                merged_rect = pymupdf.Rect(
                    min(cb["bbox"][0] for cb in cand_blocks),
                    min(cb["bbox"][1] for cb in cand_blocks),
                    max(cb["bbox"][2] for cb in cand_blocks),
                    max(cb["bbox"][3] for cb in cand_blocks),
                )
                merged_block = {"bbox": tuple(merged_rect), "lines": [{"spans": merged_spans}]}
                res = parse_equation_block(merged_block)
                if res:
                    line_indices = expanded_indices

        if res:
            for i in line_indices:
                used_indices.add(i)
            equations.append((merged_rect, res))

    return equations


def extract_clean_page_text(
    page: pymupdf.Page,
    page_num: int,
    masking_boxes: List[pymupdf.Rect],
) -> str:
    """공간 마스킹을 적용하고 1단/2단 컬럼 구조를 자동 판별하여 수식 및 본문을 정규화합니다."""
    page_width = page.rect.width
    page_height = page.rect.height
    mid_x = page_width / 2.0
    raw_blocks = page.get_text("blocks")

    # 1. 수식 블록 선행 감지 및 병합
    equations = detect_page_equations(page)

    # 2. 시각 영역 내부 텍스트 공간 마스킹
    clean_blocks = []
    for b in raw_blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        if block_type != 0 or not text.strip():
            continue

        b_rect = pymupdf.Rect(x0, y0, x1, y1)
        is_masked = False
        for m in masking_boxes:
            intersection = b_rect & m
            if not intersection.is_empty:
                overlap_ratio = intersection.get_area() / b_rect.get_area()
                if overlap_ratio > 0.35:
                    is_masked = True
                    break

        if not is_masked:
            clean_blocks.append(b)

    # 3. 수식에 흡수된 블록 분리 및 통합 단위 생성
    unified_units: List[Tuple[pymupdf.Rect, str, str]] = []
    for b in clean_blocks:
        b_rect = pymupdf.Rect(b[:4])
        b_text = b[4].strip()
        part_of_eq = False
        for eq_rect, _ in equations:
            inter = b_rect & eq_rect
            if not inter.is_empty and (inter.get_area() / b_rect.get_area()) > 0.4:
                part_of_eq = True
                break
            # 수식 주변의 미세 기호 파편(<= 6자, 시그마 첨자/분모 등) 흡수
            if len(b_text) <= 6 and abs(b_rect.y0 - eq_rect.y0) < 18 and abs(b_rect.y1 - eq_rect.y1) < 18:
                if b_rect.x0 >= eq_rect.x0 - 20 and b_rect.x1 <= eq_rect.x1 + 20:
                    part_of_eq = True
                    break
        if not part_of_eq:
            unified_units.append((b_rect, "text", b_text))

    for eq_rect, eq_str in equations:
        unified_units.append((eq_rect, "eq", eq_str))

    # 4. 1단 vs 2단 컬럼 레이아웃 적응형 감지
    crossing = sum(
        1
        for u in unified_units
        if u[1] == "text"
        and len(u[2]) > 30
        and u[0].x0 < mid_x - 20
        and u[0].x1 > mid_x + 20
    )
    total_text = sum(1 for u in unified_units if u[1] == "text" and len(u[2]) > 30)
    is_single_col = (crossing / total_text > 0.3) if total_text > 0 else True

    if is_single_col:
        # 단일 컬럼 논문: 단순 위->아래 y 좌표 정렬
        unified_units.sort(key=lambda u: u[0].y0)
    else:
        # 2단 컬럼 학술 논문: Top(Span) -> Left Column -> Right Column -> Bottom(Span)
        top, left, right, bottom = [], [], [], []
        for u in unified_units:
            r = u[0]
            is_span = r.width > (page_width * 0.65)
            if is_span and r.y0 < page_height * 0.35:
                top.append(u)
            elif is_span and r.y0 >= page_height * 0.7:
                bottom.append(u)
            elif r.x1 <= mid_x + 15:
                left.append(u)
            else:
                right.append(u)
        top.sort(key=lambda u: u[0].y0)
        left.sort(key=lambda u: u[0].y0)
        right.sort(key=lambda u: u[0].y0)
        bottom.sort(key=lambda u: u[0].y0)
        unified_units = top + left + right + bottom

    # 5. 본문 및 수식 마크다운 렌더링
    p_lines = []
    for r, kind, content in unified_units:
        if kind == "eq":
            p_lines.append(f"\n{content}\n")
        else:
            heading_info = detect_heading(content)
            if heading_info:
                level, heading_text = heading_info
                p_lines.append(f"\n{heading_text}\n")
                continue

            normalized = dehyphenate(content)
            normalized = format_bullets_and_lists(normalized)
            # 본문 내 불필요한 $$ 누수 차단
            normalized = normalized.replace("$$", r"\$\$")

            lines = normalized.splitlines()
            processed_lines = []
            for line in lines:
                if line.strip().startswith("- "):
                    processed_lines.append(line.strip())
                else:
                    if processed_lines and not processed_lines[-1].startswith("- "):
                        processed_lines[-1] += " " + line.strip()
                    else:
                        processed_lines.append(line.strip())

            p_lines.append("\n".join(processed_lines) + "\n")

    page_text = dehyphenate("\n".join(p_lines))
    return page_text


def parse_pdf_to_documents(
    source_uri: str,
    output_base_dir: Optional[Path] = None,
) -> List[Document]:
    """PDF 파일을 정밀 파싱하여 LangChain Document 리스트로 반환합니다.

    Args:
        source_uri: PDF 파일 경로 (e.g. 'data/raw/ITME.pdf')
        output_base_dir: 시각 자료 저장 경로 (기본값: 'data/extracted/{pdf_stem}')

    Returns:
        List[Document]: 각 페이지별 정제된 본문과 메타데이터가 담긴 문서 리스트
    """
    pdf_path = Path(source_uri)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {source_uri}")

    stem = pdf_path.stem
    if output_base_dir is None:
        output_dir = Path("data/extracted") / stem
    else:
        output_dir = output_base_dir / stem

    doc = pymupdf.open(str(pdf_path))
    documents = []
    page_contents = []

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]

        # 1. 시각 요소 감지 및 마스킹 박스 계산 (이미지 저장)
        visuals, masking_boxes = extract_visuals_and_mask(
            page, page_num, str(pdf_path), output_dir=output_dir
        )

        # 2. 마스킹 적용 및 본문 텍스트 정제
        clean_text = extract_clean_page_text(page, page_num, masking_boxes)
        page_contents.append((page_num, clean_text, visuals))

        # 3. 도표 표 Markdown이 존재하면 본문 하단에 함께 결합 (LLM 검색 지원)
        table_sections = []
        for v in visuals:
            if v.get("table_md"):
                table_sections.append(f"\n> **{v['caption']}**\n\n{v['table_md']}\n")
        if table_sections:
            clean_text += "\n" + "\n".join(table_sections)

        # 4. LangChain Document 객체 생성 (기존 호환 규격 준수)
        doc_obj = Document(
            page_content=clean_text.strip(),
            metadata={
                "source": str(source_uri),
                "page": page_idx,  # 0-indexed: format_docs 및 _rag_references와의 호환성 유지
                "total_pages": len(doc),
                "figures": [v["file_name"] for v in visuals if v["kind"] == "Figure"],
                "tables": [v["file_name"] for v in visuals if v["kind"] == "Table"],
            },
        )
        documents.append(doc_obj)

    # 5. 로컬 디스크에 추출된 텍스트 Markdown 파일 저장 (사용자 확인 및 아티팩트용)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        full_clean_lines = [f"# {stem} (Full Paper Clean Text)\n"]
        full_assembled_lines = [f"# {stem} (Full Paper Assembled Document)\n"]

        for p_num, p_text, visuals in page_contents:
            full_clean_lines.append(f"\n<!-- Page {p_num} -->\n")
            full_clean_lines.append(p_text)

            full_assembled_lines.append(f"\n<!-- Page {p_num} -->\n")
            full_assembled_lines.append(p_text)

            if visuals:
                full_assembled_lines.append(f"\n### Visuals on Page {p_num}\n")
                for v in visuals:
                    if v.get("table_md"):
                        full_assembled_lines.append(f"> **{v['caption']}**\n\n{v['table_md']}\n\n*Original Table*: ![{v['kind']} {v['num']}]({v['rel_path']})\n")
                    else:
                        full_assembled_lines.append(f"![{v['kind']} {v['num']}]({v['rel_path']})\n\n> **{v['caption']}**\n")

        clean_full_text = dehyphenate("\n".join(full_clean_lines))
        assembled_full_text = dehyphenate("\n".join(full_assembled_lines))

        (output_dir / "full_paper_clean.md").write_text(clean_full_text, encoding="utf-8")
        (output_dir / "full_paper_assembled.md").write_text(assembled_full_text, encoding="utf-8")
    except Exception as e:
        print(f"[WARN] 마크다운 파일 저장 실패 ({output_dir}): {e}")

    doc.close()
    return documents
