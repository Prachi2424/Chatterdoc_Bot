import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import google.generativeai as genai


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
USER_CREDENTIALS = {"admin": "1234", "chatter": "doc"}


def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index")

@st.cache_resource
def load_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def load_chat_model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

def get_chatbot_response(question, language):
    embeddings = load_embedding_model()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question in {language} using only the context below.
If the answer is not present, reply: "Answer is not available in the context."

Context:
{context}

Question:
{question}
"""

    model = load_chat_model()
    response = model.invoke(prompt)
    return response.content



def generate_quiz_from_pdf(text, language):
    model = load_chat_model()
    prompt = f"""
Create a quiz in {language} from this text:

- 3 MCQs
- 2 Fill in the blanks
- 1 Critical Thinking Question

Include Answer Key.

TEXT:
{text}
"""
    response = model.invoke(prompt)
    return response.content

# ---------- UI ----------

def main():
    st.set_page_config("ChatterDoc", "💬")

    st.markdown("## 💬 ChatterDoc — Your Multi-PDF Assistant")

    question = st.text_input("Ask a question from your PDFs")
    if question:
        with st.spinner("Thinking..."):
            answer = get_chatbot_response(question, st.session_state.get("language", "English"))
            st.markdown(answer)

    if st.button("🧠 Generate Quiz"):
        if "pdf_text" in st.session_state:
            quiz = generate_quiz_from_pdf(st.session_state["pdf_text"], st.session_state.get("language", "English"))
            st.markdown(quiz)
        else:
            st.warning("Upload and process PDFs first.")

    with st.sidebar:
        st.title("📑 Upload PDFs")
        pdf_docs = st.file_uploader("Upload", accept_multiple_files=True)

        if st.button("Process PDFs"):
            text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(text)
            get_vector_store(chunks)
            st.session_state["pdf_text"] = text
            st.success("PDFs processed successfully!")

        st.session_state["language"] = st.selectbox("Language", ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Gujarati","Punjabi"])

if __name__ == "__main__":
    main()
