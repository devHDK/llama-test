from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="Summarizer",
    page_icon="🔒",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="bllossom-8B:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


def save_message(message, role):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


prompt = ChatPromptTemplate.from_template(
    """다음은 환자와 의사 간의 진료 대화 내용이야. 대화 내용을 참고하여 아래 예시와 같은 형식으로 요약해줘. 
    Context의 정보 외에 새로운 정보를 추가하지 마. Examples 형식에 맞게 잘 작성해주면 $50 줄게

    Context: {context}

    ----------------------------------------------------------------
    Examples:
        병명: 천식
        진료 내용: 흡입기 사용 후 호흡 개선. 날씨 나쁠 때 마스크 착용, 실내 운동 및 먼지 관리 필요. 한 달 후 호흡기 검사 예정.

        병명: 소화불량
        진료 내용: 약물 복용 후 증상 호전. 규칙적인 식사와 소화 잘 되는 음식 섭취, 스트레스 관리 및 식사 후 산책 권장. 한 달 후 재진 예정.

        병명: 안구 건조증
        진료 내용: 인공눈물 사용 후 개선. 20-20-20 규칙 권장, 하루 4-5번 인공눈물 사용 가능. 실내 습도 유지 필요. 한 달 후 재진 예정.

        병명: 알레르기 피부염
        진료 내용: 연고 사용 후 개선. 알레르기 유발 요인 피하기, 보습제 사용 및 뜨거운 물 샤워 피하기 필요. 한 달 후 재진 예정.
    ----------------------------------------------------------------

    """
)


st.title("Summarizer")

st.markdown(
    """
요약 서비스입니다.
        
"""
)

send_message("요약할 내용을 입력해주세요", "ai", save=False)
paint_history()
message = st.chat_input("요약할 내용을 넣어주세요")
if message:
    send_message(message, "human")
    chain = (
        {
            "context": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    with st.chat_message("ai"):
        chain.invoke(message)
