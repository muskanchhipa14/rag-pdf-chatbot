import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Add project root to Python path (important for Streamlit Cloud)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from loader import load_and_split_pdfs
from retriever import RAGPipeline
from llm import get_llm, SUPPORTED_MODELS
from prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT,
    SUMMARY_PROMPT,
    REVISION_NOTES_PROMPT,
    QUIZ_PROMPT,
    clean_and_parse_json
)
from utils import (
    inject_custom_css,
    get_document_analytics,
    export_chat_history_to_markdown
)
# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(
    page_title="AI Study Assistant - Professional RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject ChatGPT-like Custom Style Theme
inject_custom_css()

# ----------------- SESSION STATE INITIALIZATION -----------------
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = uuid.uuid4().hex[:8]

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "previous_files" not in st.session_state:
    st.session_state.previous_files = []

if "starter_prompt" not in st.session_state:
    st.session_state.starter_prompt = ""

# ----------------- SIDEBAR -----------------
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <span style="font-size: 2.2rem;">📚</span>
        <h2 style="margin: 5px 0 0 0; font-size: 1.5rem; letter-spacing: -0.5px;">Study Workspace</h2>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 2px 0 0 0;">Advanced RAG Copilot</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Document Upload Zone
st.sidebar.markdown("### 📤 Upload Materials")
uploaded_files = st.sidebar.file_uploader(
    "Drag & drop study PDFs here",
    type="pdf",
    accept_multiple_files=True,
    key="pdf_uploader",
    label_visibility="collapsed"
)

# Check for file changes to reset session states automatically
uploaded_file_names = [f.name for f in uploaded_files] if uploaded_files else []
if uploaded_file_names != st.session_state.previous_files:
    st.session_state.previous_files = uploaded_file_names
    st.session_state.chat = []
    st.session_state.quiz_questions = None
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.summary_text = ""
    st.session_state.notes_text = ""
    st.session_state.starter_prompt = ""
    st.session_state.pipeline = None  # Reset the pipeline state!

# Initialize RAG pipeline if files exist
pipeline = None
system_ready = False
analytics = {}

if uploaded_files:
    session_data_dir = os.path.join("data", st.session_state.session_id)
    if st.session_state.pipeline is None:
        with st.sidebar.spinner("Indexing documents..."):
            # Clean session-specific data directory to prevent files accumulating
            import shutil
            if os.path.exists(session_data_dir):
                try:
                    shutil.rmtree(session_data_dir)
                except Exception:
                    pass
            os.makedirs(session_data_dir, exist_ok=True)
            
            # Save files to session directory and split into semantic chunks
            chunks = load_and_split_pdfs(uploaded_files, target_dir=session_data_dir)
            if chunks:
                st.session_state.pipeline = RAGPipeline(chunks, persist_directory=None)
                
    pipeline = st.session_state.pipeline
    if pipeline:
        system_ready = True
        analytics = get_document_analytics(pipeline.chunks)

# System Readiness Badge
if system_ready:
    st.sidebar.markdown(
        """
        <div class="status-badge status-ready">
            <span class="dot dot-green"></span>
            System Ready
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Document Manager / Details
    with st.sidebar.expander("📄 Document Manager", expanded=True):
        st.markdown(f"**Total files:** `{analytics.get('total_documents', 0)}`")
        st.markdown(f"**Total chunks:** `{analytics.get('total_chunks', 0)}`")
        st.markdown(f"**Approx. words:** `{analytics.get('total_approx_words', 0)}`")
        
        for name in analytics.get("documents_list", []):
            st.markdown(f"<span style='font-size:0.85rem; color:#cbd5e1;'>📄 {name}</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        """
        <div class="status-badge status-empty">
            <span class="dot dot-orange"></span>
            No PDF Loaded
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

# Study Material Dropdown Filter (used to filter retrieval)
if uploaded_files:
    file_options = ["All Documents"] + [f.name for f in uploaded_files]
    selected_doc_name = st.sidebar.selectbox("🎯 Target Document", file_options)
    
    # Map back to full local path of selected document
    selected_doc_path = None
    if selected_doc_name != "All Documents":
        selected_doc_path = os.path.join("data", st.session_state.session_id, selected_doc_name)
else:
    selected_doc_path = None

# Advanced Settings
with st.sidebar.expander("⚙️ AI Configuration"):
    selected_model = st.selectbox(
        "Model Type",
        options=list(SUPPORTED_MODELS.keys()),
        index=0
    )
    
    user_api_key = st.text_input(
        "Groq API Key (Override)",
        type="password",
        placeholder="Defaults to .env config",
        help="Paste your own Groq API Key if the default is rate-limited."
    )
    
    temperature = st.slider("Creativity (Temp)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# Sidebar footer details
st.sidebar.markdown(
    """
    <div style="margin-top: 60px; text-align: center; color: #64748b; font-size: 0.75rem;">
        Developed as a Portfolio Project • 2026
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- MAIN UI -----------------
st.title("📚 AI PDF Study Assistant")
st.markdown("##### Transform your textbook, slides, and resumes into custom interactive workspaces.")

# Tabs configuration
tab_chat, tab_summary, tab_quiz, tab_notes = st.tabs([
    "💬 Chat Workspace",
    "📝 Executive Summary",
    "🧠 MCQ Interactive Quiz",
    "📌 Structured Revision Notes"
])

# Get LLM instance safely
def get_current_llm():
    try:
        return get_llm(
            model_display_name=selected_model,
            api_key=user_api_key,
            temperature=temperature
        )
    except Exception as e:
        st.error(f"Configuration Error: {e}")
        st.stop()

# ----------------- TAB 1: CHAT WORKSPACE -----------------
with tab_chat:
    if not system_ready:
        st.info("💡 **Welcome!** Upload a PDF study document in the sidebar to start asking questions.")
        
        # Display sample mockup/aesthetic layout when no documents are uploaded
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-logo">🧠</div>
                <div class="welcome-title">Next-Gen Document RAG Assistant</div>
                <div class="welcome-subtitle">Uses Hybrid Keyword/Vector search + Cross-Encoder reranking for highly accurate responses.</div>
                <div style="font-weight: 500; font-size: 0.9rem; margin-bottom: 12px; color: #f1f5f9;">Features available after PDF upload:</div>
                <div style="text-align: left; max-width: 400px; margin: 0 auto; color: #cbd5e1; font-size: 0.85rem; line-height: 1.6;">
                    ✅ <b>Contextual QA</b> - Answers verified with document citations.<br>
                    ✅ <b>Summaries</b> - Clean overview, takeaways, and terminology index.<br>
                    ✅ <b>MCQ Quizzes</b> - Test yourself interactively with immediate grading.<br>
                    ✅ <b>Revision Notes</b> - Markdown revision sheets ready to export.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Chat History Container
        chat_container = st.container()
        
        # Render existing chats
        with chat_container:
            if not st.session_state.chat:
                # ChatGPT Welcome screen with quick actions
                st.markdown(
                    """
                    <div class="welcome-card">
                        <div class="welcome-logo">🤖</div>
                        <div class="welcome-title">How can I help you study?</div>
                        <div class="welcome-subtitle">Ask questions, locate specific references, or click one of the quick start cards below:</div>
                        <div class="prompt-grid">
                    """,
                    unsafe_allow_html=True
                )
                
                # Render starter cards using Streamlit columns (since HTML buttons won't send callbacks easily)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📝 Summarize the document", use_container_width=True, key="start_1"):
                        st.session_state.starter_prompt = "Provide a comprehensive summary of this document, highlighting the main points and key sections."
                        st.rerun()
                    if st.button("🔑 Key Themes & Concepts", use_container_width=True, key="start_2"):
                        st.session_state.starter_prompt = "Identify and explain the key themes, core concepts, or main ideas discussed in this document."
                        st.rerun()
                with col2:
                    if st.button("🔍 Analyze Structure & Logic", use_container_width=True, key="start_3"):
                        st.session_state.starter_prompt = "Analyze the structure, methodology, narrative style, or logic of this document."
                        st.rerun()
                    if st.button("💡 Extract Highlights & Quotes", use_container_width=True, key="start_4"):
                        st.session_state.starter_prompt = "Extract the most important quotes, key findings, or highlights from this document and explain their significance."
                        st.rerun()
                
                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                for chat in st.session_state.chat:
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(chat["q"])
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(chat["a"])
                        # Render RAG Source Citations
                        if chat.get("sources"):
                            with st.expander("🔍 View Sources & Retrieval Scores", expanded=False):
                                for doc in chat["sources"]:
                                    src_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                                    page = doc.metadata.get("page", 0) + 1
                                    score = doc.metadata.get("rerank_score", 0.0)
                                    
                                    # Similarity color scale (positive scores are green, negative are gray)
                                    score_color = "#34d399" if score > 0 else "#94a3b8"
                                    
                                    st.markdown(
                                        f"""
                                        <div class="source-card">
                                            <div class="source-header">
                                                <span>📄 {src_name} (Page {page})</span>
                                                <span style="color: {score_color}">Rerank Score: {score:.3f}</span>
                                            </div>
                                            <div class="source-snippet">"{doc.page_content.strip()}"</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    
        # Chat input handling
        # Support starter prompt click
        default_val = ""
        if st.session_state.starter_prompt:
            default_val = st.session_state.starter_prompt
            st.session_state.starter_prompt = "" # consume it
            
        user_query = st.chat_input("💬 Ask a question about your study materials...", key="chat_input")
        
        # If user used a starter card, trigger execution
        if default_val and not user_query:
            user_query = default_val

        if user_query:
            # Display user's question immediately
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)
                
            # Compute QA
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing document sources..."):
                    llm = get_current_llm()
                    docs, context = pipeline.retrieve(user_query, source_filter=selected_doc_path)
                    
                    if not docs:
                        # Fallback query general response
                        system_msg = QA_SYSTEM_PROMPT
                        user_msg = f"The user is asking a question: {user_query}. It was not found in the documents. Answer generally."
                        
                        prompt = [
                            ("system", system_msg),
                            ("user", user_msg)
                        ]
                        res = llm.invoke(prompt)
                        answer = res.content
                        sources = []
                    else:
                        system_msg = QA_SYSTEM_PROMPT
                        user_msg = QA_USER_PROMPT.format(context=context, query=user_query)
                        
                        prompt = [
                            ("system", system_msg),
                            ("user", user_msg)
                        ]
                        res = llm.invoke(prompt)
                        answer = res.content
                        sources = docs
                    
                    st.markdown(answer)
                    
                    # Output sources if any
                    if sources:
                        with st.expander("🔍 View Sources & Retrieval Scores", expanded=False):
                            for doc in sources:
                                src_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                                page = doc.metadata.get("page", 0) + 1
                                score = doc.metadata.get("rerank_score", 0.0)
                                score_color = "#34d399" if score > 0 else "#94a3b8"
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-header">
                                            <span>📄 {src_name} (Page {page})</span>
                                            <span style="color: {score_color}">Rerank Score: {score:.3f}</span>
                                        </div>
                                        <div class="source-snippet">"{doc.page_content.strip()}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
            # Save to history
            st.session_state.chat.append({
                "q": user_query,
                "a": answer,
                "sources": sources
            })
            st.rerun()

        # Chat utilities row
        if st.session_state.chat:
            st.markdown("---")
            col_clear, col_export, _ = st.columns([1, 1.2, 5])
            with col_clear:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat = []
                    st.rerun()
            with col_export:
                md_chat_logs = export_chat_history_to_markdown(st.session_state.chat)
                st.download_button(
                    "📥 Export Chat (MD)",
                    data=md_chat_logs,
                    file_name="chat_history.md",
                    mime="text/markdown",
                    use_container_width=True
                )

# ----------------- TAB 2: EXECUTIVE SUMMARY -----------------
with tab_summary:
    if not system_ready:
        st.info("💡 Upload PDF study documents in the sidebar to generate summaries.")
    else:
        st.subheader("📝 Document Summarizer")
        st.write("Generates an executive overview, key takeaways, and definitions of core terminology.")
        
        # Summary controls
        col_btn, col_scope = st.columns([2, 3])
        with col_btn:
            summary_btn = st.button("⚡ Generate Summary", use_container_width=True)
            
        with col_scope:
            summary_target = f"Selected Document: `{selected_doc_name}`" if uploaded_files else "All Loaded Documents"
            st.markdown(f"<div style='padding-top:8px;'>🎯 <i>Scope: {summary_target}</i></div>", unsafe_allow_html=True)
            
        if summary_btn:
            with st.spinner("Generating document summary..."):
                # Retrieve top chunks for summarization
                llm = get_current_llm()
                
                # Fetch chunks based on filter
                if selected_doc_path:
                    target_chunks = [c for c in pipeline.chunks if c.metadata.get("source") == selected_doc_path]
                else:
                    target_chunks = pipeline.chunks
                    
                # Take key representative chunks (e.g. up to 10 chunks from start and middle)
                subset_chunks = target_chunks[:12]
                text_content = "\n\n".join([c.page_content for c in subset_chunks])
                
                prompt = SUMMARY_PROMPT.format(text=text_content)
                res = llm.invoke(prompt)
                st.session_state.summary_text = res.content
                
        if st.session_state.summary_text:
            st.markdown("---")
            st.markdown(st.session_state.summary_text)
            
            st.markdown("---")
            st.download_button(
                "📥 Download Summary File",
                st.session_state.summary_text,
                file_name=f"summary_{selected_doc_name.replace('.pdf','')}.md",
                mime="text/markdown"
            )

# ----------------- TAB 3: MCQ INTERACTIVE QUIZ -----------------
with tab_quiz:
    if not system_ready:
        st.info("💡 Upload PDF study documents in the sidebar to take custom quizzes.")
    else:
        st.subheader("🧠 Interactive MCQ Quiz")
        st.write("Test your knowledge! Generates 5 custom multiple-choice questions from your documents.")
        
        if not st.session_state.quiz_questions:
            quiz_generate_btn = st.button("🎮 Generate Quiz", use_container_width=True)
            if quiz_generate_btn:
                with st.spinner("Creating custom quiz questions..."):
                    llm = get_current_llm()
                    
                    # Fetch representative chunks for quiz
                    if selected_doc_path:
                        target_chunks = [c for c in pipeline.chunks if c.metadata.get("source") == selected_doc_path]
                    else:
                        target_chunks = pipeline.chunks
                        
                    # Sample chunks (use a mix of chunks)
                    import random
                    sample_size = min(8, len(target_chunks))
                    # Seed random to keep it consistent or let it vary
                    sampled = random.sample(target_chunks, sample_size) if len(target_chunks) > sample_size else target_chunks
                    text_content = "\n\n".join([c.page_content for c in sampled])
                    
                    prompt = QUIZ_PROMPT.format(text=text_content)
                    
                    try:
                        res = llm.invoke(prompt)
                        # Clean and parse the response JSON
                        questions = clean_and_parse_json(res.content)
                        if isinstance(questions, list) and len(questions) > 0:
                            st.session_state.quiz_questions = questions
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_score = 0
                            st.rerun()
                        else:
                            st.error("Invalid quiz structure returned by LLM. Please try again.")
                    except Exception as e:
                        st.error(f"Failed to generate quiz: {e}")
                        st.caption("Raw output from LLM was:")
                        if 'res' in locals():
                            st.text(res.content)
                            
        # Render the interactive quiz
        if st.session_state.quiz_questions:
            questions = st.session_state.quiz_questions
            
            st.markdown("---")
            st.markdown("### 📝 Knowledge Check")
            
            # Form to wrap questions
            with st.form("quiz_form"):
                for idx, q_obj in enumerate(questions):
                    st.markdown(f"#### **Q{idx+1}. {q_obj['question']}**")
                    options = q_obj["options"]
                    
                    # Store selected answer
                    saved_answer = st.session_state.quiz_answers.get(idx, None)
                    default_idx = options.index(saved_answer) if saved_answer in options else 0
                    
                    user_select = st.radio(
                        "Choose option:",
                        options=options,
                        index=default_idx,
                        key=f"q_{idx}",
                        label_visibility="collapsed"
                    )
                    st.session_state.quiz_answers[idx] = user_select
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                # Submit form button
                submit_quiz = st.form_submit_button("Submit Quiz Answers")
                
                if submit_quiz:
                    # Grade the quiz
                    score = 0
                    for idx, q_obj in enumerate(questions):
                        selected = st.session_state.quiz_answers.get(idx)
                        if selected == q_obj["answer"]:
                            score += 1
                    st.session_state.quiz_score = score
                    st.session_state.quiz_submitted = True
                    st.rerun()
            
            # Render Quiz Feedback
            if st.session_state.quiz_submitted:
                score = st.session_state.quiz_score
                percentage = int((score / len(questions)) * 100)
                
                # Grading banners
                st.markdown("---")
                if percentage >= 80:
                    st.markdown(
                        f"""
                        <div class="quiz-score-banner" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(52, 211, 153, 0.15)); border-color: rgba(52, 211, 153, 0.4); color: #34d399;">
                            🎉 Fantastic Job! Score: {score}/{len(questions)} ({percentage}%)
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif percentage >= 50:
                    st.markdown(
                        f"""
                        <div class="quiz-score-banner" style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(251, 191, 36, 0.15)); border-color: rgba(251, 191, 36, 0.4); color: #fbbf24;">
                            👍 Good Effort! Score: {score}/{len(questions)} ({percentage}%)
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="quiz-score-banner" style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(248, 113, 113, 0.15)); border-color: rgba(248, 113, 113, 0.4); color: #f87171;">
                            📚 Keep Studying! Score: {score}/{len(questions)} ({percentage}%)
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Review Answers in Detail
                st.markdown("### 🔍 Answer Review")
                for idx, q_obj in enumerate(questions):
                    selected = st.session_state.quiz_answers.get(idx)
                    correct_ans = q_obj["answer"]
                    is_correct = selected == correct_ans
                    
                    status_symbol = "✅" if is_correct else "❌"
                    text_color = "#34d399" if is_correct else "#f87171"
                    
                    st.markdown(f"**Question {idx+1}:** {q_obj['question']}")
                    st.markdown(f"**Your Choice:** <span style='color:{text_color}; font-weight:500;'>{status_symbol} {selected}</span>", unsafe_allow_html=True)
                    
                    if not is_correct:
                        st.markdown(f"**Correct Choice:** <span style='color:#34d399; font-weight:500;'>✅ {correct_ans}</span>", unsafe_allow_html=True)
                        
                    st.info(f"💡 **Explanation:** {q_obj['explanation']}")
                    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
                    
            # Retake/Reset Button
            if st.button("🔄 Reset & Try Another Quiz", key="reset_quiz"):
                st.session_state.quiz_questions = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = 0
                st.rerun()

# ----------------- TAB 4: REVISION NOTES -----------------
with tab_notes:
    if not system_ready:
        st.info("💡 Upload PDF study documents in the sidebar to generate revision sheets.")
    else:
        st.subheader("📌 Revision Cheat-Sheets")
        st.write("Compiles high-yield revision outlines with formulas, concepts, and key highlights.")
        
        notes_btn = st.button("🚀 Generate Study Cheat-Sheet", use_container_width=True)
        
        if notes_btn:
            with st.spinner("Compiling high-yield notes..."):
                llm = get_current_llm()
                
                # Fetch representative chunks for revision notes
                if selected_doc_path:
                    target_chunks = [c for c in pipeline.chunks if c.metadata.get("source") == selected_doc_path]
                else:
                    target_chunks = pipeline.chunks
                    
                # Take key representative chunks
                subset_chunks = target_chunks[:10]
                text_content = "\n\n".join([c.page_content for c in subset_chunks])
                
                prompt = REVISION_NOTES_PROMPT.format(text=text_content)
                res = llm.invoke(prompt)
                st.session_state.notes_text = res.content
                
        if st.session_state.notes_text:
            st.markdown("---")
            st.markdown(st.session_state.notes_text)
            
            st.markdown("---")
            st.download_button(
                "📥 Download Revision Notes",
                st.session_state.notes_text,
                file_name=f"revision_notes_{selected_doc_name.replace('.pdf','')}.md",
                mime="text/markdown"
            )