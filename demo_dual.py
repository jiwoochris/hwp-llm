import streamlit as st

import olefile
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI

from langchain.schema import HumanMessage, SystemMessage

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

# from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Function to extract text from an HWP file
import olefile
import zlib
import struct

def get_hwp_text(filename):
    f = olefile.OleFileIO(filename)
    dirs = f.listdir()

    # HWP 파일 검증
    if ["FileHeader"] not in dirs or \
       ["\x05HwpSummaryInformation"] not in dirs:
        raise Exception("Not Valid HWP.")

    # 문서 포맷 압축 여부 확인
    header = f.openstream("FileHeader")
    header_data = header.read()
    is_compressed = (header_data[36] & 1) == 1

    # Body Sections 불러오기
    nums = []
    for d in dirs:
        if d[0] == "BodyText":
            nums.append(int(d[1][len("Section"):]))
    sections = ["BodyText/Section"+str(x) for x in sorted(nums)]

    # 전체 text 추출
    text = ""
    for section in sections:
        bodytext = f.openstream(section)
        data = bodytext.read()
        if is_compressed:
            unpacked_data = zlib.decompress(data, -15)
        else:
            unpacked_data = data
    
        # 각 Section 내 text 추출    
        section_text = ""
        i = 0
        size = len(unpacked_data)
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3ff
            rec_len = (header >> 20) & 0xfff

            if rec_type in [67]:
                rec_data = unpacked_data[i+4:i+4+rec_len]
                section_text += rec_data.decode('utf-16')
                section_text += "\n"

            i += 4 + rec_len

        text += section_text
        text += "\n"

    return text




def process_uploaded_file(uploaded_file):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # loader

        raw_text = get_hwp_text(uploaded_file)
        print(raw_text)
    
        
        # splitter
        text_splitter = CharacterTextSplitter(
            separator = "\r\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,
        )
        all_splits = text_splitter.create_documents([raw_text])
        
        print(all_splits)
        print("총 " + str(len(all_splits)) + "개의 passage")
        
        # storage
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
                
        return vectorstore, raw_text
    return None

def generate_response(query_text, vectorstore_1, vectorstore_2, callback):

    # retriever 1
    docs_1  = vectorstore_1.similarity_search(query_text)
    
    # retriever 2
    docs_2  = vectorstore_2.similarity_search(query_text)
    
    # generator
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callbacks=[callback])
    
    rag_prompt = [
        SystemMessage(
            content="너는 한글 문서에 대해 알려주는 \"한글이\"야. 주어진 문서를 참고하여 사용자의 질문에 답변을 해줘. 문서에 내용이 정확하게 나와있지 않으면 대답하지마."
        ),
        HumanMessage(
            content=f"질문:{query_text}\n\n2021년 문서:{docs_1[0].page_content}\n2022년 문서:{docs_2[0].page_content}"
        ),
    ]
    
    print(rag_prompt)
    
    response = llm(rag_prompt)
    
    print(response.content)

    return response.content


def generate_summarize(raw_text, callback):


    # generator
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, callbacks=[callback])
    
    rag_prompt = [
        SystemMessage(
            content="다음 나올 문서를 'Notion style'로 요약해줘. 중요한 내용만."
        ),
        HumanMessage(
            content=raw_text
        ),
    ]
    
    print(rag_prompt)
    
    response = llm(rag_prompt)
    
    print(response.content)

    return response.content


# Page title
st.set_page_config(page_title='🦜🔗 한글 hwp 문서 기반 질문 답변 챗봇')
st.title('🦜🔗 한글 hwp 문서 기반 질문 답변 챗봇')

# First file upload
uploaded_file_1 = st.file_uploader('2021년 문서를 업로드 해주세요', type='hwp')

# Second file upload
uploaded_file_2 = st.file_uploader('2022년 문서를 업로드 해주세요', type='hwp')

# File upload logic
if uploaded_file_1:
    vectorstore, raw_text = process_uploaded_file(uploaded_file_1)
    if vectorstore:
        st.session_state['vectorstore_1'] = vectorstore
        st.session_state['raw_text_1'] = raw_text
        
# File upload logic
if uploaded_file_2:
    vectorstore, raw_text = process_uploaded_file(uploaded_file_2)
    if vectorstore:
        st.session_state['vectorstore_2'] = vectorstore
        st.session_state['raw_text_2'] = raw_text

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content='안녕하세요! 저는 hwp 문서에 대한 이해를 도와주는 챗봇 \"한글이\"입니다. 어떤게 궁금하신가요?'
        )
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input("'요약'이라고 입력해보세요!"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        
        if prompt == "요약":
            response = generate_summarize(st.session_state['raw_text'], stream_handler)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
        
        else:
            response = generate_response(prompt, st.session_state['vectorstore_1'], st.session_state['vectorstore_2'], stream_handler)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
        
# streamlit run demo.py