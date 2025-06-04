import streamlit as st
import chat
from data import app_data

class ChatApp:
    def __init__(self):
        self._messages = st.session_state.setdefault("messages", [{"role": "system", "content": app_data.prompt}])
        self._chat_service = chat.ChatService()


    def _render(self):
        st.markdown(app_data.footer_html, unsafe_allow_html=True)

        for msg in self._messages[1:]:  # Skip system message
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


    def _chat(self):
        user_input = st.chat_input("메시지를 입력하세요...")
        if user_input:
            self._messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("메시지 작성 중..."):
                    reply = self._chat_service.get_reply(user_input)
                    st.markdown(reply)
                    self._messages.append({"role": "assistant", "content": reply})


    def run(self):
        self._render()
        self._chat()


if __name__ == "__main__":
    app = ChatApp()
    app.run()