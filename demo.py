import streamlit as st

import olefile
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

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
        
        print("AAA")
        
        from openai import OpenAI
        
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "다음 나올 문서를 요약해줘."},
                {"role": "user", "content": raw_text[:3000]}
            ]
        )
        
        print(response)
        
        exit()
        
        # splitter
        text_splitter = CharacterTextSplitter(
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,
        )
        all_splits = text_splitter.create_documents([raw_text])
        

        # storage
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        
        return vectorstore
    return None

def generate_response(query_text, vectorstore, callback):

    # retriever
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
    
    # generator
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callbacks=[callback])
    
    rag_prompt = PromptTemplate.from_template(
        "주어진 문서를 참고하여 사용자의 질문에 답변을 해줘.\n\n질문:{question}\n\n문서:{context}"
    )
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    def log_and_invoke(query):
        docs = retriever.get_relevant_documents(query)
        print(f"Retrieved {len(docs)} documents for query: {query}")
        print(docs)
        
        return rag_chain.invoke(query)

    response = log_and_invoke(query_text)

    return response


# Page title
st.set_page_config(page_title='🦜🔗 문서 기반 질문 답변 챗봇')
st.title('🦜🔗 문서 기반 질문 답변 챗봇')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='hwp')

# File upload logic
if uploaded_file:
    vectorstore = process_uploaded_file(uploaded_file)
    if vectorstore:
        st.session_state['vectorstore'] = vectorstore

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content='안녕하세요! 저는 문서를 기반으로 답변해주는 챗봇입니다. 어떤게 궁금하신가요?'
        )
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        response = generate_response(uploaded_file, prompt, stream_handler)
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response)
        )
        
# streamlit run demo.py