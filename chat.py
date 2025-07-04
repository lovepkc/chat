import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from openai import OpenAI

import streamlit as st
from dotenv import load_dotenv

import db

load_dotenv()
#OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

documents = [Document(page_content=text) for text in db.TEXTS]
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db_vector = FAISS.from_documents(docs, embedding)
retriever = db_vector.as_retriever()

client = OpenAI(api_key=OPENAI_API_KEY)


def get_reply(user_input):
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role":"system", "content":db.PROMPT},
            {"role":"user", 
             "content":f"다음 아버지가 주로 사용하던 문장을 바탕으로, 따뜻하고 인자한 말투로 대답해줘.:\n\n{retriever} {user_input}"}
        ],
        temperature=0.7
    )
    
    return completion.choices[0].message.content
        