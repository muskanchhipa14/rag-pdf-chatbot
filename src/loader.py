import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def save_uploaded_file(uploaded_file, target_dir="data"):
    """
    Saves a streamlit uploaded file to the local target directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def load_and_split_pdfs(file_paths_or_uploaded_files, chunk_size=800, chunk_overlap=150, target_dir="data"):
    """
    Loads one or more PDFs, splits them into recursive character-based chunks,
    and returns the list of document chunks.
    """
    documents = []
    
    for item in file_paths_or_uploaded_files:
        # Check if the item is a streamlit UploadedFile or a file path string
        if hasattr(item, "name") and hasattr(item, "getbuffer"):
            file_path = save_uploaded_file(item, target_dir=target_dir)
        else:
            file_path = str(item)
            
        if not os.path.exists(file_path):
            continue
            
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    if not documents:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks
