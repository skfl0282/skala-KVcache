"""prompts/*.txt 템플릿을 PromptTemplate으로 불러오는 공용 유틸리티.
"""

from langchain_core.prompts import PromptTemplate


def load_prompt(path: str) -> PromptTemplate:
    """prompts/ 폴더의 .txt 템플릿 파일을 읽어 PromptTemplate으로 변환합니다."""
    with open(path, encoding="utf-8") as f:
        template = f.read()
    return PromptTemplate.from_template(template)
