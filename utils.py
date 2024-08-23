from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from model import get_all_contents
import os
from config import OPENAI_API_KEY

# 환경 변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "kita project 2nd team webapp"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

VECTORSTORE_PATH = "vectorstore.faiss"

def initialize_vectorstore():
    sections = get_all_contents()

    # RecursiveCharacterTextSplitter 사용
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50 
    )
    
    # 섹션을 분할하여 문서 리스트 생성
    documents = []
    for section in sections:
        splits = text_splitter.split_text(section)
        documents.extend([Document(page_content=split) for split in splits])

    # 문서에 대한 임베딩 생성 및 벡터스토어 초기화
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 벡터스토어를 로컬에 저장
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def get_relevant_sections(user_input):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

    # 가장 유사한 벡터 검색
    #retriever = vectorstore.as_retriever(search_type="similarity")

    # 가장 유사한 결과를 찾으면서 결과 간의 중복을 최소화. 유사성과 다양성을 동일하게 고려
    retriever = vectorstore.as_retriever(search_type="mmr", k=5, lambda_mult=0.5)
    
    relevant_documents = retriever.invoke(user_input)
    relevant_sections = "\n".join([doc.page_content for doc in relevant_documents])

    return relevant_sections

def get_ai_response(user_input, chat_history):

    template = """
        ## 당신은 근로자들을 상담해주는 근로자 상담 관련 챗봇입니다.
        - 법률에 관한 db는 사용자가 관련된 법이나 법 조항을 물어보고 꼭 필요할때 사용하세요.
        - 당신의 역할은 근로기준법을 모르는 사람에게 관련 법과 이에 관한 제도들을 설명해주는 것입니다.
        - 답변이 정확하지 않을 가능성이 높을 경우 잘 모른다고 답변해야 됩니다. 부정확한 답변은 하지 않는게 좋습니다.
        - 질문이 구체적이라면 그 방법도 구체적으로 알려주는것이 좋습니다.
        - 유저의 질문에 정보가 많이 없을때는 한번 더 질문을 해서 더 구체적인 답변을 주는것이 좋습니다. 예를 들면 월급이나 수당관련되서는 날짜를 다시 물어본 후 계산을 해줄지 한번 더 물어본다음 계산을 해주면 원하는 답변을 해줄 수 있어요.
        - 한 가지 주제에 대한 질문에 답변 후, 사용자가 놓쳤을 수 있는 관련 주제에 대해 추가 질문을 제안해 주세요.
        - 답변에 대해서는 최대한 상세하게 알려주세요.
        - 금액 산정에 관한 질문에는 돈의 성격에 따라서 세금이나, 수당, 공제 등 여러가지 변수들이 있을 수 있음으로 성격에 맞게 이를 명시하여 주세요.
        - 법률적 조언과 일반적인 정보 제공의 차이를 명확히 해주세요. 법률에 관한 정보를 줄 수는 있으나, 전문 변호사나 상담 기관에 문의할 것을 권장해 주세요.

        1. 

        문서 내용: {context}

        대화 내용:
        {chat_history}

        질문: {human_input}

        응답:
    """
    # 템플릿을 사용하여 프롬프트 생성
    custom_prompt = PromptTemplate.from_template(template)

    # 컨텍스트를 가져오는 부분 (사용자 정의 함수라고 가정)
    context = get_relevant_sections(user_input)

    # LLM 생성
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0.2)

    # 체인 생성
    chain = custom_prompt | llm

    # 기존 대화 내역 포함
    inputs = {"context": context, "human_input": user_input, "chat_history": chat_history}
    response = chain.invoke(inputs)

    # 대화 내역 저장
    chat_history = chat_history + f"\nUser: {user_input}\nAI: {response.content}"

    return response.content, chat_history