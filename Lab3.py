import streamlit as st
from openai import OpenAI

st.title("💬 Lab 3: Chatbot with memory")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a helpful assistant talking to a 10 year old.
Explain everything in simple words, short sentences, no jargon.
After you answer a question, always end by asking "Do you want more info?"
If the user says yes, give more detail on the same topic, then ask "Do you want more info?" again.
If the user says no, say okay and ask what else you can help with."""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # buffer: keep only the last 2 user messages and their replies
    buffer = st.session_state.messages[-4:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + buffer

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )
        answer = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": answer})