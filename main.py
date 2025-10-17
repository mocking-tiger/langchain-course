from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate  # 프롬프트 템플릿 생성
from langchain_openai import ChatOpenAI  # OpenAI 모델 사용

load_dotenv()


def main():
    print("main함수를 실행합니다.")
    information = """
일론 머스크는 미국의 기업인, 행정가.

남아공에서 태어나 캐나다를 거쳐 미국에 정착했으며, 미국·캐나다·남아공 삼중국적을 보유하고 있다.

테슬라, 스페이스X, 스타링크, 뉴럴링크, xAI 홀딩스 등 다수의 첨단 기업을 보유 및 운영하면서 2020년대 들어 세계에서 가장 부유한 기업인 반열에 등극했다. 실용 중심적인 경영, 이를 아우르는 미래 지향적 비전 등이[37] 그의 특징으로 평가받고 있다.

인류 역사상 개인 순자산 4000억달러(572조원)를 넘은 최초의 인물이다. 머스크가 세운 또 다른 회사인 우주 기업 스페이스X의 기업가치 상승도 머스크의 순자산 확대에 크게 기여했다. 블룸버그는 머스크의 순자산이 4392억달러(약628조원)에 이른다고 추산했다.
    """

    summary_template = """
    {information}
    위 정보에 대해
    1. 짧은 요약
    2. 두 가지 흥미로운 사실
    을 출력하세요.
    """

    summary_prompt_template = PromptTemplate(template=summary_template, input_variables=["information"])

    llm = ChatOpenAI(model="gpt-5", temperature=0)

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)



if __name__ == "__main__":
    main()
