"""
보고서 분량 맞춤 (PDF MAX_PAGES쪽 이내)

report_gen이 쓴 보고서를 PDF로 저장했을 때 MAX_PAGES쪽을 넘으면, LLM으로 내용을 압축한 뒤
다시 저장하기를 MAX_CONDENSE_ATTEMPTS번까지 반복한다.

- save_report_pdf가 먼저 글자 크기를 줄여 보고(10.5pt -> 9pt), 그래도 넘칠 때만 압축한다.
- 압축은 필수 장/소제목, 수치와 출처 표기, REFERENCE 항목을 유지하고 중복 서술과 긴 표만 줄인다.
  새로운 내용을 추가하지 않으므로 품질 평가(Groundedness 등) 기준을 해치지 않는다.
- 목표 분량은 "현재 글자 수 x (MAX_PAGES / 현재 쪽수) x 여유 비율"로 잡고, 시도할수록 여유를 더 둔다.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.report_pdf import MAX_PAGES, save_report_pdf
from agents.worker_utils import PLANNER_MODEL

MAX_CONDENSE_ATTEMPTS = 3
# 시도마다 목표 분량에 곱하는 여유 비율 (LLM이 목표보다 길게 쓰는 경향이 있어 점점 더 짧게 요구)
TARGET_MARGINS = (0.9, 0.8, 0.7)

condense_prompt = ChatPromptTemplate.from_template(
    """당신은 기술 평가 보고서의 편집자입니다.
아래 [보고서]는 PDF로 {pages}쪽이라 분량 제한({max_pages}쪽)을 넘습니다.
내용의 의미와 근거는 유지한 채 전체 분량을 약 {target_chars}자 이내(현재 {current_chars}자)로 줄여 보고서 전체를 다시 쓰세요.

[반드시 유지할 것]
- 마크다운 장/소제목 구조: SUMMARY, 1. 분석 배경, 2. 기술 선정, 3. 기술 개요, 4. 관점별 평가
  (4.1 기술성숙도 관점, 4.2 시장성 관점, 4.3 이해관계자 관점, 4.4 도메인 적용 관점, 4.5 종합의견),
  5. 시사점, 6. 한계점, REFERENCE. 제목의 이름과 순서를 바꾸지 마세요.
- 수치, 기업명, 사례와 그 출처 표기(원문 파일/페이지, URL). 남기는 문장의 근거 표기는 지우지 마세요.
- TRL 등급과 "공개 정보 기반 추정"이라는 명시, 종합의견의 관점 간 상충 지점, "근거 부족" 표시.
- 중립적인 서술 (우열 판정이나 추천을 새로 넣지 마세요).
- REFERENCE의 항목 표기 (새 항목을 만들지 말고, 남기는 항목은 그대로 옮기세요).

[줄이는 방법]
- 여러 절에서 반복되는 설명(특히 SUMMARY, 시사점, 종합의견 사이의 중복)을 한 번만 쓰기
- 긴 문단은 핵심 문장 위주로 압축, 부연 설명과 수식어 줄이기
- 표는 꼭 필요한 행·열만 남기고, 표와 본문이 같은 내용을 반복하면 한쪽만 남기기
- 새로운 내용, 수치, 출처를 추가하지 마세요.

보고서 본문만 출력하세요 (설명이나 머리말 없이).

[보고서]
{report}"""
)
condense_llm = init_chat_model(PLANNER_MODEL, model_provider="openai", temperature=0)
condense_chain = condense_prompt | condense_llm | StrOutputParser()


def fit_report_to_pages(report: str, path: str, max_pages: int = MAX_PAGES) -> tuple[str, int]:
    """보고서를 path에 PDF로 저장하고, max_pages를 넘으면 압축해 다시 저장한다.

    (최종 보고서 마크다운, PDF 쪽수)를 반환한다. 압축을 다 시도해도 넘치면 가장 짧은
    결과로 저장하고 경고를 출력한다 (분량 초과 여부는 quality_eval의 '분량' 항목이 다시 판정).
    압축 LLM 호출이 실패하면 그 시점의 보고서를 그대로 둔다.
    """
    pages = save_report_pdf(report, path, max_pages)
    best_report, best_pages = report, pages

    for attempt, margin in enumerate(TARGET_MARGINS[:MAX_CONDENSE_ATTEMPTS], start=1):
        if best_pages <= max_pages:
            break
        target_chars = int(len(best_report) * max_pages / best_pages * margin)
        print(f"  [분량 맞춤] {best_pages}쪽 > {max_pages}쪽 -> {attempt}차 압축 (목표 약 {target_chars}자)")
        try:
            condensed = condense_chain.invoke(
                {
                    "report": best_report,
                    "pages": best_pages,
                    "max_pages": max_pages,
                    "current_chars": len(best_report),
                    "target_chars": target_chars,
                }
            ).strip()
        except Exception as e:
            print(f"[WARN] 보고서 압축 실패, 현재 보고서를 유지합니다: {e}")
            break
        if not condensed:
            break
        pages = save_report_pdf(condensed, path, max_pages)
        if pages <= best_pages:  # 압축이 오히려 길어진 경우는 버린다
            best_report, best_pages = condensed, pages

    if best_pages > max_pages:
        print(f"[WARN] 압축 후에도 {best_pages}쪽으로 분량 제한({max_pages}쪽)을 넘습니다: {path}")
    # 마지막으로 렌더링한 PDF가 best_report가 아닐 수 있으므로 최종본으로 다시 저장한다
    best_pages = save_report_pdf(best_report, path, max_pages)
    return best_report, best_pages
