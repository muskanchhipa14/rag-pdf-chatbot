import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# ---------------- IMPORTS ----------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

if "quiz_text" not in st.session_state:
    st.session_state.quiz_text = ""

if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""

# ---------------- UI ----------------
st.set_page_config(
    page_title="AI Study Assistant",
    layout="wide"
)

st.title("📚 AI Study Assistant")

st.write(
    "Upload study material and ask questions, generate summaries, and learn faster."
)

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True,
    key="pdf_upload"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📚 Study Workspace")

st.sidebar.markdown("### Uploaded Documents")

if uploaded_files:
    for file in uploaded_files:
        st.sidebar.write("📄 " + file.name)

st.sidebar.markdown("---")

st.sidebar.info("""
Features:
✅ Ask Questions
✅ Summaries
✅ Quizzes
✅ Revision Notes
""")

selected_doc = st.sidebar.selectbox(
    "📄 Choose Study Material",
    ["All Documents"] + [file.name for file in uploaded_files] if uploaded_files else ["All Documents"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Recent Questions")

chat_history = st.session_state.get("chat", [])

if chat_history:
    for item in chat_history[-5:][::-1]:
        st.sidebar.write("• " + item["q"])
else:
    st.sidebar.caption("No questions asked yet")

# ---------------- LLM ----------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

# ---------------- PIPELINE ----------------
@st.cache_resource
def load_pipeline(uploaded_files):

    documents = []

    for uploaded_file in uploaded_files:

        save_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(save_path)
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        return [], None, None, None

    texts = [c.page_content for c in chunks]
    tokenized = [t.lower().split() for t in texts]

    bm25 = BM25Okapi(tokenized)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return chunks, bm25, vectorstore, reranker


chunks, bm25, vectorstore, reranker = [], None, None, None

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        chunks, bm25, vectorstore, reranker = load_pipeline(uploaded_files)

system_ready = (
    vectorstore is not None
    and bm25 is not None
    and reranker is not None
    and len(chunks) > 0
)

# ---------------- INPUT ----------------
query = st.text_input("💬 Ask something from your PDFs:")
topic = st.text_input("🔍 Enter topic (optional)")

summary_button = st.button("📝 Generate Summary")
quiz_button = st.button("🧠 Generate Quiz")
notes_button = st.button("📌 Generate Revision Notes")

# ---------------- VALIDATION ----------------
if query:
    if len(query.strip()) < 3:
        st.warning("⚠ Too short query")
        query = ""
    elif sum(c.isalpha() for c in query) / len(query) < 0.4:
        st.warning("⚠ Not meaningful query")
        query = ""

# ---------------- SUMMARY ----------------
if summary_button:

    if not system_ready:
        st.warning("Please upload PDF files first.")
        st.stop()

    with st.spinner("Generating summary..."):

        topic_query = expand_query(topic or "key concepts")
        docs = vectorstore.similarity_search(topic_query, k=5)
        text = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are an AI assistant.

Generate:
1. Summary
2. Key points
3. Important concepts

Content:
{text}
"""

        res = llm.invoke(prompt)

        st.session_state.summary_text = res.content

        st.text_area("Summary", res.content, height=250)

        st.download_button(
            "Download Summary",
            res.content,
            file_name="summary.txt"
        )
#----------- quiz 
if quiz_button:

    if not system_ready:
        st.warning("Upload PDFs first.")
        st.stop()

    with st.spinner("Generating quiz..."):

        docs = vectorstore.similarity_search("key concepts important questions", k=6)

        text = "\n\n".join([d.page_content for d in docs])

        quiz_prompt = f"""
Create a quiz from the following content.

Rules:
- 5 MCQs
- 4 options each
- mark correct answer
- simple exam style

Content:
{text}
"""

        response = llm.invoke(quiz_prompt)

        st.markdown("## 🧠 Quiz")
        st.write(response.content)
# ---------------- NOTES ----------------
if notes_button:

    if not system_ready:
        st.warning("Upload PDF first.")
        st.stop()

    text = "\n\n".join([c.page_content for c in chunks[:5]])

    prompt = f"""
Create concise revision notes:

{text}
"""

    res = llm.invoke(prompt)

    st.session_state.notes_text = res.content

    st.markdown("## 📌 Revision Notes")
    st.success(res.content)

    st.download_button(
        "Download Notes",
        res.content,
        file_name="revision_notes.txt"
    )

# ---------------- QA SYSTEM ----------------
def expand_query(q):
    if not q:
        return ""

    return f"""
{q}
skills projects experience education clubs achievements resume
"""
def retrieve(query):

    expanded_query = expand_query(query)

    vector_docs = vectorstore.similarity_search(expanded_query, k=5)

    tokenized = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized)

    top_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:5]

    bm25_docs = [chunks[i] for i in top_idx]

    candidates = vector_docs + bm25_docs

    if not candidates:
        return [], ""

    pairs = [[query, d.page_content] for d in candidates]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, candidates), reverse=True)
    final_docs = [d for _, d in scored[:3]]

    context = "\n\n".join([d.page_content for d in final_docs])

    return final_docs, context


if query and system_ready:

    with st.spinner("Thinking..."):

        docs, context = retrieve(query)

        if not docs:
            answer = llm.invoke(
                f"No match in PDF. Answer generally: {query}"
            ).content
            sources = []
        else:
            prompt = f"""
Answer only using context:

{context}

Question: {query}
"""

            answer = llm.invoke(prompt).content
            sources = docs

        st.session_state.chat.append({
            "q": query,
            "a": answer,
            "sources": sources
        })

        st.rerun()

# ---------------- CHAT UI ----------------
for chat in reversed(st.session_state.chat):

    st.markdown("### 👤 You")
    st.write(chat["q"])

    st.markdown("### 🤖 AI")
    st.success(chat["a"])

    with st.expander("📌 Sources"):
        for d in chat["sources"]:
            st.write(d.page_content[:300])
            st.caption(d.metadata.get("source", "unknown"))

    st.divider()

# ---------------- EMPTY STATE ----------------
if not query and not summary_button and not notes_button and not quiz_button:
    st.info("Upload PDFs and start asking questions 🚀")