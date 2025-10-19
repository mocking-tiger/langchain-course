from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

# tavily에서 만든 검색 툴
tools = [TavilySearch()]

# llm 모델 사용(gpt-5는 stop인수를 지원하지 않음)
llm = ChatOpenAI(
    model="gpt-4", temperature=0
)
structured_llm = llm.with_structured_output(AgentResponse)

# react 프롬프트 사용
react_prompt = hub.pull("hwchase17/react")

# 출력 파서 생성
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

# 프롬프트 템플릿 생성
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
# ).partial(format_instructions=output_parser.get_format_instructions())
).partial(format_instructions='')

# react 에이전트 생성
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)

# 에이전트 실행자
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# RunnableLambda를 사용하여 출력 추출
extract_output = RunnableLambda(lambda x: x["output"])
# parse_output = RunnableLambda(lambda x: output_parser.parse(x))

chain = agent_executor | extract_output | structured_llm


def main():
    print("main함수를 실행합니다.")

    result = chain.invoke(
        {
            "input": "langchain을 사용하여 서울 지역의 linkedin의 프론트엔드 개발자 채용 공고 3개를 검색하고 세부정보를 나열해라."
        }
    )
    print(result)


if __name__ == "__main__":
    main()
