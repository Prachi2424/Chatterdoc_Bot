
import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#Authentication
USER_CREDENTIALS = {"admin": "1234", "chatter": "doc"}

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        with st.spinner("Logging in..."):
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

def get_conversational_chain(language):
    prompt_template = f"""
    Answer the question in **{language}** using only the context below.
    If the answer is not present, reply: "Answer is not available in the context."

    Context:
    {{context}}

    Question:
    {{question}}

    Answer:
    """
    model = load_chat_model()
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_chatbot_response(question, language):
    embeddings = load_embedding_model()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)
    chain = get_conversational_chain(language)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

#Quiz generator

def generate_quiz_from_pdf(text, language):
    model = load_chat_model()
    prompt = f"""
    You are a helpful assistant. Based on the following text from a PDF, create a short quiz in **{language}** with:
    - 3 Multiple Choice Questions (MCQs)
    - 2 Fill-in-the-blanks
    - 1 Critical Thinking Question

    After all the questions, create an **Answer Key** section with correct answers.

    TEXT:
    {text}
    """
    response = model.invoke(prompt)
    return response.text()

#  Main App

def main():
    st.set_page_config(page_title="ChatterDoc", page_icon="💬")

    st.markdown("<h2 style='text-align:left;'>💬 Your multi-PDF assistant — “ChatterDoc at your service”</h2>", unsafe_allow_html=True)
    st.markdown(" 🔍 Ask a Question from the PDF Files")
   
    pdf_prompt = st.text_input("Type you question here...", key="pdf_query")
    if pdf_prompt:
        with st.chat_message("user"):
            st.markdown(pdf_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching your PDF..."):
                response = get_chatbot_response(pdf_prompt, st.session_state.get("selected_language", "English"))
                st.markdown(response)
 # Quiz Button
    if st.button("🧠 Generate Quiz from PDF"):
        if "pdf_text" in st.session_state:
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz_from_pdf(st.session_state["pdf_text"], st.session_state.get("selected_language", "English"))
                st.success("Here’s a quiz based on your PDF:")
                st.markdown(quiz)
        else:
            st.warning("Please upload and process a PDF first.")
            

    # Sidebar for uploading and language selection
    with st.sidebar:
        st.title("📑 Upload PDF")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
                st.session_state["pdf_text"] = text
                st.success("✅ PDFs processed successfully!")

        language = st.selectbox("🌐 Select Language", ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Gujarati"])
        st.session_state["selected_language"] = language
        st.caption(f"Currently selected: {language}")

if __name__ == "__main__":
    main()
