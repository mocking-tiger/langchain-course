import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader # 텍스트 파일을 로드하는 모델
from langchain_text_splitters import CharacterTextSplitter # 텍스트를 작은 단위로 나누는 모델(토큰 초과 방지)
from langchain_openai import OpenAIEmbeddings # 텍스트를 벡터로 변환하는 모델
from langchain_pinecone import PineconeVectorStore # 벡터 데이터베이스를 사용하는 모델

# Windows 터미널에서 UTF-8 출력 지원
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

def main():
    print("main 함수 실행")
    loader = TextLoader("./practice1.txt", encoding="utf-8")
    documents = loader.load()

    # 텍스트를 작은 단위로 나누기
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))

    # 작은 단위를 pinecone에 저장
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ['INDEX_NAME']
    )
    print("ingestion complete")

if __name__ == "__main__":
    main()
