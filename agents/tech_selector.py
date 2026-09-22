"""
기술 선정 에이전트
담당: __________ (TODO: 담당자 배정)

설계서 B. 설계 - 기술선정 방식은 Human 기반(조가 직접 선정)이므로 이 노드는
LLM 호출 없이, 조에서 확정한 기술명/도메인을 State에 기록만 한다.
"""

from graph.state import GraphState

# 조에서 Human 기반으로 확정한 기술 (설계서 B 참고)
DEFAULT_TECH_SW = "DeepSeek-V2 (MLA)"
DEFAULT_TECH_HW = "ITME (Inference Tiered Memory Expansion)"
DEFAULT_DOMAIN = "데이터센터/클라우드 서빙"


def select_technology(state: GraphState):
    """SW/HW 진영 선정 기술명과 평가 도메인을 State에 write한다.

    TODO: 현재는 상수로 하드코딩되어 있음. CLI 인자나 설정 파일에서 읽어오도록 확장 가능.
    """
    print("\n==== [SELECT TECHNOLOGY] ====\n")

    tech_sw = state.get("tech_sw") or DEFAULT_TECH_SW
    tech_hw = state.get("tech_hw") or DEFAULT_TECH_HW
    domain = state.get("domain") or DEFAULT_DOMAIN
    print(f"SW={tech_sw} / HW={tech_hw} / domain={domain}")

    return {
        "tech_sw": tech_sw,
        "tech_hw": tech_hw,
        "domain": domain,
        "messages": [("system", f"[기술 선정] SW={tech_sw}, HW={tech_hw}, domain={domain}")],
    }
