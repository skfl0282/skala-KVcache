"""
실행 스크립트
담당: __________ (TODO: 담당자 배정 - 통합/실행 담당)

사용법:
    python app.py
    python app.py --tech-sw "DeepSeek-V2 (MLA)" --tech-hw "ITME" --domain "데이터센터/클라우드"
"""

import argparse

from dotenv import load_dotenv

load_dotenv(override=True)

from agents.tech_selector import DEFAULT_DOMAIN, DEFAULT_TECH_HW, DEFAULT_TECH_SW
from graph.build_graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV Cache 최적화 기술 다관점 평가 Agentic RAG")
    parser.add_argument("--tech-sw", default=DEFAULT_TECH_SW, help="SW 진영 선정 기술명")
    parser.add_argument("--tech-hw", default=DEFAULT_TECH_HW, help="HW 진영 선정 기술명")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="평가 도메인")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = build_graph()

    initial_state = {
        "tech_sw": args.tech_sw,
        "tech_hw": args.tech_hw,
        "domain": args.domain,
        "data_limited": [],
    }

    final_state = app.invoke(initial_state)

    print("=" * 60)
    print("실행 완료. 최종 보고서는 outputs/report.md 에 저장되었습니다.")
    print("=" * 60)
    print(final_state.get("report", "(report가 비어 있습니다)"))


if __name__ == "__main__":
    main()
