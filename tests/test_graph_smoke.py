"""
그래프 스모크 테스트: Orchestrator-Workers 전문 워커 기반 그래프 조립 및 배선 검증
"""

from langgraph.graph import END
from langgraph.types import Send

from agents.quality_eval import route_after_review
from graph.build_graph import assign_workers, build_graph
from graph.state import GraphState, ReportState


def test_graph_structure():
    """그래프에 모든 필수 노드가 등록되어 있는지 검증"""
    app = build_graph()
    assert app is not None
    node_keys = list(app.get_graph().nodes.keys())

    assert "orchestrator" in node_keys
    assert "tech_research" in node_keys
    assert "market_eval" in node_keys
    assert "stakeholder_eval" in node_keys
    assert "domain_eval" in node_keys
    assert "synthesizer" in node_keys
    assert "report_gen" in node_keys
    assert "quality_eval" in node_keys


def test_assign_workers():
    """assign_workers가 tasks로부터 Send 목록을 정상 생성하는지 검증"""
    mock_tasks = {
        "t1": {
            "task_id": "t1",
            "worker": "tech_research",
            "perspective": "기술성숙도",
            "instruction": "inst",
            "search_query": "q",
            "target": "sw",
            "status": "pending",
            "attempts": 0,
            "error": None,
        },
        "t2": {
            "task_id": "t2",
            "worker": "market_eval",
            "perspective": "시장성",
            "instruction": "inst",
            "search_query": "q",
            "target": None,
            "status": "pending",
            "attempts": 0,
            "error": None,
        },
    }
    mock_state = {
        "tech_sw": "DeepSeek-V2 (MLA)",
        "tech_hw": "ITME",
        "domain": "클라우드",
        "run_id": "test",
        "tasks": mock_tasks,
        "decisions": [],
        "results": {},
        "synthesis": "",
    }
    sends = assign_workers(mock_state)
    assert len(sends) == 2
    for s in sends:
        assert isinstance(s, Send)
        assert s.node in ("tech_research", "market_eval")
        assert s.arg["tech_sw"] == "DeepSeek-V2 (MLA)"


def test_route_after_review():
    """quality_eval 결정에 따른 라우팅 분기 검증"""
    # 1. revise 결정 -> report_gen
    state_revise = {
        "decisions": [{"node": "quality_eval", "action": "revise", "task_ids": [], "reason": "미달"}]
    }
    assert route_after_review(state_revise) == "report_gen"

    # 2. accept 결정 -> END
    state_accept = {
        "decisions": [{"node": "quality_eval", "action": "accept", "task_ids": [], "reason": "통과"}]
    }
    assert route_after_review(state_accept) == END

    # 3. stop 결정 -> END
    state_stop = {
        "decisions": [{"node": "quality_eval", "action": "stop", "task_ids": [], "reason": "상한 도달"}]
    }
    assert route_after_review(state_stop) == END
