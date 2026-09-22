"""
그래프 스모크 테스트
담당: __________ (TODO: 담당자 배정)

각 agents/*.py의 내부 로직이 아직 STUB이어도, 그래프 배선 자체가
START부터 END까지 끊김 없이 연결되는지 확인하는 최소한의 테스트.
"""

from graph.build_graph import build_graph


def test_graph_runs_end_to_end():
    app = build_graph()

    initial_state = {
        "tech_sw": "DeepSeek-V2 (MLA)",
        "tech_hw": "ITME",
        "domain": "데이터센터/클라우드",
        "data_limited": [],
    }

    final_state = app.invoke(initial_state)

    # 세 fan-out 브랜치 결과가 모두 채워졌는지
    assert final_state.get("market_eval")
    assert final_state.get("stakeholder_eval")
    assert final_state.get("domain_eval")

    # fan-in 이후 노드들도 채워졌는지
    assert final_state.get("synthesis")
    assert final_state.get("report")


if __name__ == "__main__":
    test_graph_runs_end_to_end()
    print("OK: 그래프가 START부터 END까지 정상적으로 연결되어 있습니다.")
