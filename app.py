"""
실행 스크립트: Orchestrator-Workers 전문 워커 기반 KV Cache 기술 평가 시스템

사용법:
    python app.py
    python app.py --tech-sw "DeepSeek-V2 (MLA)" --tech-hw "ITME" --domain "데이터센터/클라우드 서빙"
    python app.py --graph               # 그래프 실행 없이 outputs/graph.png만 생성
"""

import argparse
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# LangSmith 추적 환경변수 기본값 설정
os.environ.setdefault("LANGSMITH_PROJECT", "SKALA-KVCACHE")

from agents.tech_selector import DEFAULT_DOMAIN, DEFAULT_TECH_HW, DEFAULT_TECH_SW
from graph.build_graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV Cache 최적화 기술 다관점 평가 Orchestrator-Workers 시스템"
    )
    parser.add_argument("--tech-sw", default=DEFAULT_TECH_SW, help="SW 진영 선정 기술명")
    parser.add_argument("--tech-hw", default=DEFAULT_TECH_HW, help="HW 진영 선정 기술명")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="평가 대상 도메인")
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

    run_id = uuid.uuid4().hex[:8]
    print("=" * 70)
    print(" [KV Cache Optimization Evaluation] Orchestrator-Workers Pipeline")
    print(f" - Run ID         : {run_id}")
    print(f" - SW Technology  : {args.tech_sw}")
    print(f" - HW Technology  : {args.tech_hw}")
    print(f" - Target Domain  : {args.domain}")
    print("=" * 70)

    initial_state = {
        "tech_sw": args.tech_sw,
        "tech_hw": args.tech_hw,
        "domain": args.domain,
        "run_id": run_id,
        "tasks": {},
        "results": {},
        "decisions": [],
    }

    config = {
        "configurable": {"thread_id": f"thread-{run_id}"},
        "run_name": f"kv-cache-eval-report-{run_id}",
        "tags": ["app", "quality-loop", "specialized-workers"],
    }

    final_state = app.invoke(initial_state, config=config)

    review = final_state.get("review") or {}
    tasks = final_state.get("tasks", {})
    done_count = sum(1 for t in tasks.values() if t.get("status") == "done")
    failed_count = sum(1 for t in tasks.values() if t.get("status") in ("failed", "excluded"))

    print("\n" + "=" * 70)
    print(" 파이프라인 실행 완료 요약")
    print("=" * 70)
    print(f" - Run ID                : {run_id}")
    print(f" - 완료된 태스크 수      : {done_count}개 (제외/실패: {failed_count}개)")
    print(f" - 보고서 작성 횟수(회)  : {final_state.get('revision', 1)}회")
    if review:
        print(f" - 최종 품질 평가 판정   : {'통과 (PASSED)' if review.get('passed') else '미달 (FAILED)'}")
        if review.get("error"):
            print(f"   (오류: {review['error']})")
        for c in review.get("criteria", []):
            print(f"   {'[통과]' if c['passed'] else '[미달]'} {c['name']}: {c['comment']}")
    print(" - 저장된 최종 보고서    : outputs/report_orchestrator_workers.md")
    print("=" * 70 + "\n")

    report_content = final_state.get("report", "(보고서가 비어 있습니다)")
    print("=== [최종 보고서 미리보기 (상위 500자)] ===")
    print(report_content[:500] + ("..." if len(report_content) > 500 else ""))


if __name__ == "__main__":
    main()
