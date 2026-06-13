import streamlit as st
import io
import os

def inject_custom_css():
    """
    Injects professional CSS to transform the Streamlit page.
    Features:
      - Custom Google Fonts (Inter & Plus Jakarta Sans)
      - Glassmorphism containers
      - ChatGPT-like sidebar & chat margins
      - Pulse status indicator animations
      - Beautiful button hover effects
      - Styled scrollbars
      - Hidden default Streamlit headers/footers for a standalone app feel
    """
    st.markdown(
        """
        <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        /* Apply Fonts safely without breaking icon webfonts */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            font-family: 'Inter', sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 700 !important;
        }
        
        /* Clean header & footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        .stDeployButton {
            display: none !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0d0f14 !important;
            border-right: 1px solid #1e293b !important;
        }
        
        /* Sidebar Title and elements */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
        }
        
        /* Main background */
        .stApp {
            background-color: #0f172a;
            color: #f1f5f9;
        }
        
        /* Chat Input Styling */
        [data-testid="stChatInput"] {
            border-radius: 12px !important;
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1) !important;
        }
        
        /* Custom buttons styling */
        .stButton>button {
            border-radius: 8px !important;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 2px 4px rgba(99, 102, 241, 0.2) !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
            background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
        }
        
        /* Secondary Action Buttons (e.g. download, clear) */
        .stDownloadButton>button, button[key*="secondary"] {
            border-radius: 8px !important;
            background-color: #1e293b !important;
            color: #f1f5f9 !important;
            border: 1px solid #334155 !important;
            font-weight: 500 !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        .stDownloadButton>button:hover {
            background-color: #334155 !important;
            color: white !important;
            border-color: #475569 !important;
        }
        
        /* Source Citation Cards */
        .source-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 8px;
            margin-bottom: 8px;
            backdrop-filter: blur(10px);
            font-size: 0.85rem;
        }
        
        .source-header {
            font-weight: 600;
            color: #818cf8;
            margin-bottom: 4px;
            display: flex;
            justify-content: space-between;
        }
        
        .source-snippet {
            color: #cbd5e1;
            font-style: italic;
        }
        
        /* Pulse Status Indicator */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .status-ready {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.2);
        }
        
        .status-empty {
            background-color: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.2);
        }
        
        .dot {
            height: 8px;
            width: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        
        .dot-green {
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse-green 2s infinite;
        }
        
        .dot-orange {
            background-color: #f59e0b;
            box-shadow: 0 0 8px #f59e0b;
            animation: pulse-orange 2s infinite;
        }
        
        @keyframes pulse-green {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        
        @keyframes pulse-orange {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(245, 158, 11, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        }
        
        /* Chat Welcome Screen */
        .welcome-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            max-width: 650px;
            margin: 40px auto;
            backdrop-filter: blur(12px);
        }
        
        .welcome-logo {
            font-size: 3rem;
            margin-bottom: 15px;
        }
        
        .welcome-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 8px;
            color: #f8fafc;
        }
        
        .welcome-subtitle {
            font-size: 0.95rem;
            color: #94a3b8;
            margin-bottom: 24px;
        }
        
        .prompt-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 15px;
        }
        
        .prompt-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 12px 16px;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s ease;
            font-size: 0.85rem;
            color: #cbd5e1;
        }
        
        .prompt-card:hover {
            border-color: #6366f1;
            background: rgba(99, 102, 241, 0.1);
            color: #f1f5f9;
        }
        
        /* Styled tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #1e293b;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: nowrap;
            border-radius: 6px 6px 0px 0px;
            background-color: transparent;
            color: #94a3b8;
            font-weight: 600;
            padding: 0 16px;
            transition: all 0.2s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #f1f5f9;
            background-color: rgba(255, 255, 255, 0.03);
        }
        
        .stTabs [aria-selected="true"] {
            color: #818cf8 !important;
            border-bottom: 2px solid #6366f1 !important;
        }
        
        /* Scrollbars styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #334155;
        }
        
        /* Custom avatar borders */
        .stChatMessage [data-testid="stChatMessageAvatar"] {
            background-color: transparent !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Quiz elements */
        .quiz-container {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
            margin-bottom: 15px;
        }
        
        .quiz-title {
            color: #f8fafc;
            margin-bottom: 12px;
        }
        
        .quiz-score-banner {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(52, 211, 153, 0.1) 100%);
            border: 1px solid rgba(52, 211, 153, 0.3);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            font-weight: 600;
            color: #34d399;
            font-size: 1.1rem;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def get_document_analytics(chunks):
    """
    Computes analytics of loaded documents to present on a resume-worthy dashboard.
    """
    total_chunks = len(chunks)
    if total_chunks == 0:
        return {}
        
    total_characters = sum(len(c.page_content) for c in chunks)
    avg_chunk_size = total_characters // total_chunks
    
    # Extract unique source names
    sources = set()
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        sources.add(os.path.basename(src))
        
    return {
        "total_documents": len(sources),
        "documents_list": list(sources),
        "total_chunks": total_chunks,
        "avg_chunk_length": avg_chunk_size,
        "total_approx_words": int(total_characters / 5)
    }

def export_chat_history_to_markdown(chat_history):
    """
    Converts session chat history list into a markdown file stream.
    """
    output = io.StringIO()
    output.write("# AI Study Assistant Chat History\n\n")
    for chat in chat_history:
        output.write(f"### 👤 User:\n{chat['q']}\n\n")
        output.write(f"### 🤖 Assistant:\n{chat['a']}\n\n")
        if chat.get("sources"):
            output.write("**Sources Reference:**\n")
            for doc in chat["sources"]:
                src_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", 0) + 1
                output.write(f"- File: {src_name} (Page {page})\n")
        output.write("---\n\n")
    return output.getvalue()
