import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ---------------- INIT ----------------
st.title("📄 AI PDF Chatbot (RAG System)")

# Load PDF once
loader = PyPDFLoader("data/Resume_MuskanChhipa.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

texts = [c.page_content for c in chunks]
tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- CHAT UI ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask something from your PDF:")

if user_input:

    vector_docs = vectorstore.similarity_search(user_input, k=3)

    tokenized_q = user_input.lower().split()
    bm25_scores = bm25.get_scores(tokenized_q)

    top_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:3]

    bm25_docs = [chunks[i] for i in top_idx]

    candidates = vector_docs + bm25_docs

    pairs = [[user_input, d.page_content] for d in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    final_docs = [d for _, d in scored[:3]]

    context = "\n\n".join([d.page_content for d in final_docs])

    prompt = f"""
    Answer based only on context:

    {context}

    Question: {user_input}
    """

    response = llm.invoke(prompt)

    st.session_state.chat.append((user_input, response.content))

# ---------------- DISPLAY CHAT ----------------
for q, a in st.session_state.chat:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")
    st.markdown("---")