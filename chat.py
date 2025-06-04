import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

from data import app_data


class ChatService:
    def __init__(self, model="gpt-3.5-turbo", k=2):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._model = model
        self._k = k
        self._docs = app_data.texts
        self._system_prompt = app_data.prompt


    def _retrieve_docs(self, user_input):
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(self._docs + [user_input])
        query_vec = doc_vectors[-1]
        doc_vectors = doc_vectors[:-1]
        similarities = cosine_similarity(query_vec, doc_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:self._k]

        return [self._docs[i] for i in top_indices]


    def _build_prompt(self, user_input):
        retrieved = self._retrieve_docs(user_input)
        context = "\n".join(retrieved)

        return f"다음 정보를 참고하여 대답하세요:\n\n{context}\n\n{user_input}"


    def get_reply(self, user_input):
        full_prompt = self._build_prompt(user_input)

        response = openai.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": full_prompt}
            ]
        )

        return response.choices[0].message.content
