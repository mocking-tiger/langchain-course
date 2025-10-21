import os  # 환경변수 접근을 위한 모듈
from dotenv import load_dotenv  # .env 파일에서 환경변수를 로드하는 모듈
from langchain_core.prompts import PromptTemplate  # 프롬프트 템플릿을 생성하는 클래스
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI의 LLM과 임베딩 모델
from langchain_pinecone import PineconeVectorStore  # Pinecone 벡터 데이터베이스 연동 클래스
from langchain import hub  # LangChain Hub에서 미리 만들어진 프롬프트를 가져오는 모듈
from langchain.chains.combine_documents import create_stuff_documents_chain  # 검색된 문서들과 프롬프트를 결합하여 LLM 체인을 만드는 함수
from langchain.chains.retrieval import create_retrieval_chain  # 벡터 DB 검색과 LLM 응답을 연결하는 RAG 체인을 만드는 함수
load_dotenv()  # .env 파일의 환경변수들을 os.environ에 로드

def main():
    print("main 함수 실행")

    embeddings = OpenAIEmbeddings()  # OpenAI 임베딩 모델 초기화 (텍스트를 벡터로 변환)
    llm = ChatOpenAI()  # OpenAI LLM 초기화 (기본 모델: gpt-3.5-turbo)

    query = "what is Pinecone in machine learning?"  # 질문 정의
    chain = PromptTemplate.from_template(template=query) | llm  # 프롬프트 템플릿과 LLM을 파이프라인으로 연결
    # result = chain.invoke(input={})  # LLM에게 질문 전달하여 일반적인 답변 생성 (RAG 없이)
    # print(result.content)  # LLM의 응답 내용 출력

    vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)  # Pinecone 벡터 DB 연결 (저장된 문서 임베딩에 접근)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")  # LangChain Hub에서 RAG용 프롬프트 템플릿 가져오기
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)  # 검색된 문서들을 프롬프트에 결합하여 LLM에 전달하는 체인 생성
    retrival_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)  # 벡터 DB 검색 + 문서 결합 + LLM 응답을 하나의 RAG 체인으로 연결
    result = retrival_chain.invoke(input={"input": query})  # RAG 체인 실행: 질문과 유사한 문서 검색 후 LLM이 답변 생성
    print(result)  # RAG 결과 출력 (검색된 문서 정보와 최종 답변 포함)

if __name__ == "__main__":  # 이 파일이 직접 실행될 때만 아래 코드 실행
    main()  # main 함수 호출
