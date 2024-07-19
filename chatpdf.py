from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
import absl.logging
import logging
import streamlit as st
import tempfile

# 제목 설정
st.title('ChatPDF')
st.write('---')

# 환경 변수 설정
os.environ["GRPC_VERBOSITY"] = "NONE"

# 로그 초기화
absl.logging.set_verbosity('info')
absl.logging.use_python_logging()

# 기본 로그 수준 설정
logging.basicConfig(level=logging.INFO)

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
api_key = os.getenv('GOOGLE_API_KEY')

# API 키를 사용하여 genai 구성
genai.configure(api_key=api_key)

# 파일 업로드
uploaded_file = st.file_uploader('PDF 파일을 업로드 하세요')
st.write('---')

def pdf_to_document(uploaded_file):
    # 임시 디렉토리 생성
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    
    # 업로드된 파일을 임시 파일로 저장
    with open(temp_filepath, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    # PDF 로더 초기화 및 페이지 로드
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    
    return pages

# 업로드된 파일이 있을 때 동작
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    
    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # 한 청크당 글자 수
        chunk_overlap=20,  # 청크 간 겹치는 글자 수
        length_function=len,  # 길이 계산 함수
        is_separator_regex=False,  # 분리자 정규 표현식 사용 여부
    )

    # 임베딩 모델 설정
    go = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

    # PDF 텍스트 분할 및 저장
    texts = text_splitter.split_documents(pages)
    texts = [doc.page_content for doc in texts]

    # 벡터 저장소 생성
    vectorstore = FAISS.from_texts(texts, embedding=go)

    # 검색기 생성
    retriever = vectorstore.as_retriever()

    # 프롬프트 템플릿 설정
    template = """Answer the question in sentences based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # AI 답변 생성 모델 설정
    model = genai.ChatModel(model="gemini-pro")

    # 체인 설정 및 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # 질문 입력 및 답변 생성
    st.header('ChatPDF에게 질문해보세요!!')
    question = st.text_input('질문을 입력하세요')
    if st.button('질문하기'):
        with st.spinner('답변하는 중...'):
            answer = chain.invoke(question)
            st.write(answer)
