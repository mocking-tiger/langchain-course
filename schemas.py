from typing import List
from pydantic import (
    BaseModel,
    Field,
)  # Pydantic의 기본 모델(BaseModel)과 필드 메타데이터 도구(Field)


class Source(BaseModel):  # 출처 정보를 표현하는 데이터 모델
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):  # 에이전트의 응답을 표현하는 데이터 모델
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )
