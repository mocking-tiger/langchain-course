from dotenv import load_dotenv  # .env 파일에서 환경변수를 로드하는 모듈
from langchain_community.document_loaders import PyPDFLoader  # PDF 파일을 읽어 Document 객체로 변환하는 로더
from langchain_text_splitters import CharacterTextSplitter  # 문서를 작은 청크(chunk)로 분할하는 스플리터
from langchain_openai import OpenAIEmbeddings, OpenAI  # OpenAI의 임베딩 모델과 LLM
from langchain_community.vectorstores import FAISS  # 로컬에서 사용 가능한 벡터 데이터베이스 (Facebook AI Similarity Search)
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
load_dotenv()  # .env 파일의 환경변수들을 os.environ에 로드

def main():
    print("main 함수 실행")
    pdf_path = './startup-bible.pdf'  # 로드할 PDF 파일의 경로 지정
    loader = PyPDFLoader(file_path=pdf_path)  # PDF 로더 초기화
    documents = loader.load()  # PDF 파일을 읽어서 Document 객체 리스트로 변환 (각 페이지가 하나의 Document)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separator="\n")  # 텍스트 스플리터 설정 (최대 1000자, 청크 간 20자 중복, 줄바꿈으로 구분)
    docs = text_splitter.split_documents(documents)  # Document 리스트를 더 작은 청크들로 분할 (벡터 DB 저장 및 검색 효율성을 위해)
    # print(docs)  # 분할된 청크들 출력

    embeddings = OpenAIEmbeddings()  # OpenAI 임베딩 모델 초기화 (텍스트를 벡터로 변환)
    vectorstore = FAISS.from_documents(docs, embeddings)  # 분할된 문서들을 임베딩하여 FAISS 벡터 DB에 저장
    vectorstore.save_local("faiss_index_react")  # 생성된 벡터 DB를 로컬 디스크에 저장 (faiss_index_react 폴더에 저장됨)

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)  # 저장된 FAISS 벡터 DB를 로컬에서 불러오기 (allow_dangerous_deserialization은 pickle 역직렬화 허용)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")  # LangChain Hub에서 RAG용 QA 프롬프트 템플릿 가져오기
    combine_docs_chain = create_stuff_documents_chain(OpenAI(), retrieval_qa_chat_prompt)  # 검색된 문서들을 프롬프트와 결합하여 LLM에 전달하는 체인 생성
    retrieval_chain = create_retrieval_chain(new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)  # 벡터 DB 검색 + 문서 결합 + LLM 응답을 연결하는 RAG 체인 생성
    res = retrieval_chain.invoke({"input": "고객획득비용이란 무엇인가요?"})  # RAG 체인 실행: 질문과 유사한 문서를 검색하고 LLM이 답변 생성
    print(res["answer"])  # 생성된 답변 출력 (res는 dict 형태이며 "answer" 키에 최종 답변이 담김)

if __name__ == "__main__":  
    main()
