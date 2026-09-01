import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

st.title("📄 Lab 2 - Document summarizer")
st.write("Upload a document and pick how you want it summarized.")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

language = st.sidebar.selectbox(
    "Output language",
    ["English", "Spanish", "French", "Chinese"],
)

summary_type = st.sidebar.selectbox(
    "Summary type",
    [
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points",
    ],
)

advanced = st.sidebar.checkbox("Use advanced model")
effort = "medium" if advanced else "none"
st.sidebar.write(f"Using: gpt-5.5 (effort: {effort})")

uploaded_file = st.file_uploader("Upload a document", type=("txt", "md", "pdf"))

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        document = "".join(page.extract_text() or "" for page in reader.pages)
    else:
        document = uploaded_file.read().decode()

    instruction = f"{summary_type}. Write the summary in {language}."

    try:
        response = client.responses.create(
            model="gpt-5.5",
            input=f"Here's a document: {document}\n\n---\n\n{instruction}",
            reasoning={"effort": effort},
        )
        st.write(response.output_text)
    except Exception as e:
        st.error(f"Something went wrong: {e}")