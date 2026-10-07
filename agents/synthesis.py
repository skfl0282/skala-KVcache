"""
평가 종합 에이전트 (Synthesizer)
모든 전문 워커의 병렬 조사 결과를 집계하고, 관점 간 일치/상충 지점을 병렬 서술하여 종합 의견을 생성한다.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.orchestrator import VIEWPOINTS
from agents.worker_utils import PLANNER_MODEL
from graph.state import Decision, GraphState, merge_tasks

synth_prompt = ChatPromptTemplate.from_template(
    """당신은 여러 관점의 평가 결과를 종합하는 분석가입니다.
아래는 여러 워커가 병렬로 조사한 기술성숙도 관점, 시장성 관점, 이해관계자 관점, 도메인 적용 관점의 결과입니다.
이를 종합해 최종 보고서의 "종합 의견"이 될 내용을 작성하세요.

- SW 기술: {tech_sw}
- HW 기술: {tech_hw}
- 평가 도메인: {domain}

[원칙]
- 객관적이고 중립적으로 쓰세요. 어느 기술이 더 낮다고 판정하거나 특정 기술을 추천하지 마세요.
- 관점 간 상충 지점이 명시적으로 드러나야 합니다. 서로 어긋나는 평가를 평균 내거나 뭉개서 하나의 결론으로
  만들지 말고, "어느 관점에서는 A, 다른 관점에서는 B"라는 형태로 그대로 드러내세요.
- 아래 조사 결과에 있는 내용만 쓰고, 새로운 수치나 사례를 추가하지 마세요.
- 근거가 부족해 판단할 수 없는 부분은 "판단 유보"로 남기세요.

[관점별 조사 결과]
{joined}

[자료 한계]
{limitations}

[출력 형식]
- 관점별 핵심 결론: (네 관점 각각 1~2문장, 병렬 서술)
- 관점 간 일치하는 지점: (어느 관점들이 무엇에 대해 일치하는지, 관점 이름을 밝혀서)
- 관점 간 상충하는 지점: (어느 관점과 어느 관점이 무엇에 대해 어긋나는지, 각 관점의 근거와 함께)
- 판단 유보 사항: (근거 부족으로 결론을 내릴 수 없는 부분)
- 자료 한계: (위 [자료 한계]에 있는 항목만. 없으면 "없음")"""
)

synth_llm = init_chat_model(PLANNER_MODEL, model_provider="openai", temperature=0)
synth_chain = synth_prompt | synth_llm | StrOutputParser()


def collect_limitations(tasks: dict) -> list[str]:
    """제외되었거나 일부 자료가 실패한 task를 보고서 한계점 재료로 뽑는다."""
    return [
        f"{task['perspective']} ({'제외' if task['status'] == 'excluded' else '일부 자료 실패'}): "
        f"{task['error']}"
        for task in tasks.values()
        if task.get("error")
    ]


def collect_references(state: GraphState) -> list[str]:
    """종합에 쓰인 task들의 출처를 순서 유지 + 중복 제거해서 모은다."""
    refs = [
        ref
        for task_id, result in state["results"].items()
        if state["tasks"][task_id]["status"] == "done"
        for ref in result["references"]
    ]
    return list(dict.fromkeys(refs))


def synthesizer(state: GraphState) -> dict:
    # --- Fall-back 판정: 실패한 task는 제외, 일부 자료만 실패한 task는 계속 ---
    updates: dict = {}
    decisions: list[Decision] = []
    for task_id, task in state["tasks"].items():
        if task["status"] == "failed":
            updates[task_id] = {"status": "excluded"}
            decisions.append(
                {
                    "node": "synthesizer",
                    "action": "exclude",
                    "task_ids": [task_id],
                    "reason": f"{task['attempts']}회 시도 후 결과 없음({task['error']})",
                }
            )
        elif task.get("error"):
            decisions.append(
                {
                    "node": "synthesizer",
                    "action": "continue",
                    "task_ids": [task_id],
                    "reason": f"일부 자료 실패, 남은 자료로 계속({task['error']})",
                }
            )
    tasks = merge_tasks(state["tasks"], updates)

    joined = "\n\n".join(
        f"### [{VIEWPOINTS[task['worker']]}] {task['perspective']}\n{state['results'][task_id]['content']}"
        for task_id, task in tasks.items()
        if task["status"] == "done"
    )
    limitations = collect_limitations(tasks)
    synthesis = synth_chain.invoke(
        {
            "tech_sw": state["tech_sw"],
            "tech_hw": state["tech_hw"],
            "domain": state["domain"],
            "joined": joined or "(조사 결과 없음)",
            "limitations": "\n".join(f"- {item}" for item in limitations) or "없음",
        }
    )
    return {"synthesis": synthesis, "tasks": updates, "decisions": decisions}
