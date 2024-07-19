# 버전 이슈 해결^^

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import absl.logging
import logging
import streamlit as st
import tempfile
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

st.title('ChatPDF')
st.write('---')

# 환경 변수 설정
os.environ["GRPC_VERBOSITY"] = "NONE"

# 로그 초기화
absl.logging.set_verbosity('info')
absl.logging.use_python_logging()

# 기본 로그 수준 설정
logging.basicConfig(level=logging.INFO)

# 환경 변수에서 API 키 가져오기
api_key = os.getenv('GOOGLE_API_KEY')

# API 키를 사용하여 genai 구성
genai.configure(api_key=api_key)

# 파일 업로드
uploaded_file = st.file_uploader('')
st.write('---')

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, 'wb') as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    pass
    
    # 운수 좋은 날.pdf 파일의 경로 설정
    #unsu = 'C:\\cording\\chatpdf\\unsu.pdf'

    # '운수 좋은 날.pdf' 파일을 가져오기 위한 PDF 로더 설정
    #loader = PyPDFLoader(file_path=unsu)

    # PDF 파일을 페이지별로 분할하여 로드
    #pages = loader.load_and_split()

    # 텍스트 분할기 설정: 주어진 설정에 따라 텍스트를 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # 한 청크당 글자 수
        chunk_overlap=20,  # 청크 간 겹치는 글자 수
        length_function=len,  # 글자 길이를 계산하는 함수
        is_separator_regex=False,  # 분리자를 정규 표현식으로 사용할지 여부
    )

    # 임베딩 모델 설정: 주어진 모델을 사용하여 텍스트 임베딩 생성
    go = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

    # PDF 파일의 텍스트를 분할하여 'texts'에 저장
    texts = text_splitter.split_documents(pages)

    # 'Document' 객체에서 텍스트 문자열 추출
    texts = [doc.page_content for doc in texts]

    # 벡터 저장소 생성: 분할된 텍스트를 임베딩하여 벡터 저장소에 저장
    vectorstore = FAISS.from_texts(texts, embedding=go)

    # 검색기 생성: 벡터 저장소를 검색할 수 있도록 설정
    retriever = vectorstore.as_retriever()

    # 질문에 대한 답변을 생성하기 위한 프롬프트 템플릿 설정
    template = """Answer the question it in sentences based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # AI 답변 생성 모델 설정
    model = ChatGoogleGenerativeAI(model="gemini-pro")

    # 전체 체인을 묶어서 완성: 입력 질문을 받아서 답변 생성까지의 과정 정의
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    st.header('ChatPDF에게 질문해보세요!!')
    question = st.text_input('질문을 입력하세요')
    if st.button('질문하기'):
        with st.spinner('답변하는 중...'):
            # 체인을 실행하여 질문에 대한 답변 생성
            answer = chain.invoke(question)
            st.write(answer)
