# from typing import List, Union // Union 타입 필요없음
from typing import List

from dotenv import load_dotenv
# from langchain.agents.format_scratchpad import format_log_to_str // 함수 호출 방식에서는 추론 과정을 공개하지 않음
# from langchain.agents.output_parsers import ReActSingleInputOutputParser // JSON을 반환하므로 텍스트를 파싱할 필요 없음
# from langchain_core.agents import AgentAction, AgentFinish // 마지막인지 아닌지 아규먼트로 받기 때문에 필요없음
# from langchain_core.prompts import PromptTemplate // 도구를 선택하는 로직을 LLM공급자(open ai, google등)가 처리함.
# from langchain_core.tools import Tool, render_text_description, tool // 도구에 대한 텍스트 설명도 필요없음
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler

load_dotenv()
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # 만약을 대비해 비알파벳 문자 제거하기
    return len(text)
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("main함수 실행")
    tools = [get_text_length]

    #################################################ReAct용 레거시 코드#########################################################
    # template = """
    # Answer the following questions as best you can. You have access to the following tools:
    # {tools}
    
    # Use the following format:
    
    # Question: the input question you must answer
    # Thought: you should always think about what to do
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action
    # ... (this Thought/Action/Action Input/Observation can repeat N times)
    # Thought: I now know the final answer
    # Final Answer: the final answer to the original input question
    
    # Begin!
    
    # Question: {input}
    # Thought: {agent_scratchpad}
    # """

    # prompt = PromptTemplate.from_template(template=template).partial(
    #     tools=render_text_description(tools),
    #     tool_names=", ".join([t.name for t in tools]),
    # )


    # intermediate_steps = [] // 따로 tool_message가 저장되므로 필요없음
    # agent = (
    #     {
    #         "input": lambda x: x["input"],
    #         "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    #     }
    #     | prompt
    #     | llm
    #     | ReActSingleInputOutputParser()
    # )

    # agent_step = ""
    # while not isinstance(agent_step, AgentFinish):
    #     agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
    #         {
    #             "input": "What is the length of the word: DOG",
    #             "agent_scratchpad": intermediate_steps,
    #         }
    #     )
    #     print(agent_step)

    #     if isinstance(agent_step, AgentAction):
    #         tool_name = agent_step.tool
    #         tool_to_use = find_tool_by_name(tools, tool_name)
    #         tool_input = agent_step.tool_input

    #         observation = tool_to_use.func(str(tool_input))
    #         print(f"{observation=}")
    #         intermediate_steps.append((agent_step, str(observation)))

    # if isinstance(agent_step, AgentFinish):
    #     print(agent_step.return_values)
    ##############################################################################################################################

    llm = ChatOpenAI(
        temperature=0,
        # stop=["\nObservation", "Observation"], // LLM의 출력 포맷에 의존하지 않아도 됨
        callbacks=[AgentCallbackHandler()],
    )

    # tools와 llm을 바인딩
    llm_with_tools = llm.bind_tools(tools)

    # 에이전트가 반복될 때마다 이 목록에 추가됨
    messages = [HumanMessage(content="What is the length of the word: DOG")]

    while True:
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = getattr(ai_message, "tool_calls", None) or []

        # tool_calls가 빈 배열일 경우 도구를 호출하지 않음
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call은 일반적으로 키가 있는 사전(객체)이다: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            # Continue loop to allow the model to use the observations
            continue

        # No tool calls -> final answer
        print(ai_message.content)
        break