from typing import List, Union
from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description, Tool
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad import format_log_to_str
load_dotenv()

@tool  # tool 데코레이터를 사용하여 도구로 등록
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""  # agent가 이 설명을 읽고 도구를 사용할 지 판단하므로 꼭 작성해야 함
    text = text.strip("'\n").strip('"')
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == "__main__":
    print("main함수 실행\n")

    # tools 리스트에 도구를 추가
    tools = [get_text_length]

    # ReAct 프롬프트 템플릿
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}    

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # LLM 설정 (Observation이 나오면 멈춤)
    llm = ChatOpenAI(temperature=0, stop=["\nObservation","Observation"])

    intermediate_steps = []

    # ReAct agent 체인 구성 - 수동으로 구성하여 내부 동작 원리 학습
    agent = ({"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])} | prompt | llm | ReActSingleInputOutputParser())

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input": "What is the length in characters of the text DOG?", "agent_scratchpad": intermediate_steps}
    )

    print(f"step1: {agent_step}")

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"Observation: {observation}")
        intermediate_steps.append((agent_step, str(observation)))

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input": "What is the length in characters of the text DOG?", "agent_scratchpad": intermediate_steps}
    )

    print(f"step2: {agent_step}")
