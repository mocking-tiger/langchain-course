from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# tavily에서 만든 검색 툴
tools = [TavilySearch()]

# llm 모델 사용
llm = ChatOpenAI(model="gpt-4", temperature=0) # llm 모델 사용

# react 프롬프트 사용
react_prompt = hub.pull("hwchase17/react")

# react 에이전트 생성
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

# 에이전트 실행자
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    print("main함수를 실행합니다.")

    result = agent_executor.invoke({"input": "langchain을 사용하여 서울 지역의 linkedin의 프론트엔드 개발자 채용 공고 3개를 검색하고 세부정보를 나열해라."})
    print(result)


if __name__ == "__main__":
    main()
