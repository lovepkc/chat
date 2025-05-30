import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from db import app_data

class ChatService:
    def __init__(self, model="gpt-3.5-turbo", k=2):
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        self.model = model
        self.k = k
        self.docs = app_data.texts
        self.system_prompt = app_data.prompt


    def retrieve_docs(self, query):
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(self.docs + [query])
        query_vec = doc_vectors[-1]
        doc_vectors = doc_vectors[:-1]
        similarities = cosine_similarity(query_vec, doc_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:self.k]

        return [self.docs[i] for i in top_indices]


    def build_prompt(self, user_input):
        retrieved = self.retrieve_docs(user_input)
        context = "\n".join(retrieved)

        return f"다음 정보를 참고하여 대답하세요:\n\n{context}\n\n{user_input}"


    def get_reply(self, user_input):
        full_prompt = self.build_prompt(user_input)

        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_prompt}
            ]
        )

        return response.choices[0].message.content
