"""
실행 스크립트
담당: __________ (TODO: 담당자 배정 - 통합/실행 담당)

사용법:
    python app.py
    python app.py --tech-sw "DeepSeek-V2 (MLA)" --tech-hw "ITME" --domain "데이터센터/클라우드"
    python app.py --graph               # 그래프 실행 없이 outputs/graph.png만 생성
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv(override=True)

# LangSmith 추적: API 키와 추적 on/off는 .env에서 읽고, 프로젝트 이름만 여기서 정한다
# (.env에 LANGSMITH_PROJECT가 있으면 그 값을 쓴다)
os.environ.setdefault("LANGSMITH_PROJECT", "SKALA-KVCACHE")

from agents.tech_selector import DEFAULT_DOMAIN, DEFAULT_TECH_HW, DEFAULT_TECH_SW
from graph.build_graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV Cache 최적화 기술 다관점 평가 Agentic RAG")
    parser.add_argument("--tech-sw", default=DEFAULT_TECH_SW, help="SW 진영 선정 기술명")
    parser.add_argument("--tech-hw", default=DEFAULT_TECH_HW, help="HW 진영 선정 기술명")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="평가 도메인")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="그래프를 실행하지 않고 build_graph() 결과를 outputs/graph.png로 렌더링만 하고 종료",
    )
    return parser.parse_args()


def render_graph(app) -> None:
    """build_graph()로 컴파일된 그래프를 실제 노드/엣지 그대로 outputs/graph.png에 렌더링한다."""
    os.makedirs("outputs", exist_ok=True)
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("outputs/graph.png", "wb") as f:
        f.write(png_bytes)
    print("outputs/graph.png 저장 완료")


def main() -> None:
    args = parse_args()

    app = build_graph()

    if args.graph:
        render_graph(app)
        return

    initial_state = {
        "tech_sw": args.tech_sw,
        "tech_hw": args.tech_hw,
        "domain": args.domain,
        "data_limited": [],
    }

    final_state = app.invoke(
        initial_state, config={"run_name": "kv-cache-eval", "tags": ["app"]}
    )

    print("=" * 60)
    print("실행 완료. 최종 보고서는 outputs/report.md 에 저장되었습니다.")
    print("=" * 60)
    print(final_state.get("report", "(report가 비어 있습니다)"))


if __name__ == "__main__":
    main()
