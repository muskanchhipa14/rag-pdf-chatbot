import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_reranker_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def expand_query(q):
    """
    Expands the query to focus on typical resume/study contents if needed,
    or returns the query as-is if it's already specific.
    """
    if not q:
        return ""
    return f"{q}\nskills projects experience education clubs achievements resume study topics concepts"

class RAGPipeline:
    def __init__(self, chunks, persist_directory=None):
        self.chunks = chunks
        self.texts = [c.page_content for c in chunks]
        
        # Initialize BM25 for lexical search
        tokenized_texts = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Initialize HuggingFace Embeddings (Cached globally)
        self.embeddings = get_embeddings_model()
        
        # Initialize Chroma vectorstore with a unique collection name to prevent doc pollution across uploads
        import uuid
        collection_name = f"col_{uuid.uuid4().hex[:12]}"
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Initialize CrossEncoder for reranking candidates (Cached globally)
        self.reranker = get_reranker_model()

    def retrieve(self, query, k_semantic=5, k_lexical=5, final_k=3, source_filter=None):
        """
        Executes a hybrid search (semantic + lexical), merges results,
        and reranks them using a CrossEncoder to return the top `final_k` documents.
        Supports filtering by a specific source document path.
        """
        if not self.chunks:
            return [], ""
            
        # 1. Semantic Search
        expanded_q = expand_query(query)
        search_filter = {"source": source_filter} if source_filter else None
        semantic_docs = self.vectorstore.similarity_search(
            expanded_q, 
            k=k_semantic, 
            filter=search_filter
        )
        
        # 2. Lexical Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        top_lexical_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k_lexical + 10] # get a few extra to account for source filtering
        
        lexical_docs = []
        for idx in top_lexical_indices:
            doc = self.chunks[idx]
            if source_filter and doc.metadata.get("source") != source_filter:
                continue
            lexical_docs.append(doc)
            if len(lexical_docs) >= k_lexical:
                break
        
        # 3. Merge candidates (deduplicate based on page content & metadata)
        seen_contents = set()
        candidates = []
        for doc in (semantic_docs + lexical_docs):
            doc_id = (doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("page", 0))
            if doc_id not in seen_contents:
                seen_contents.add(doc_id)
                candidates.append(doc)
                
        if not candidates:
            return [], ""
            
        # 4. Rerank candidates with CrossEncoder
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # Zip scores and candidates, sort descending by score
        scored_candidates = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        
        # Extract top final_k documents and record their rerank scores
        final_docs = []
        for score, doc in scored_candidates[:final_k]:
            # Add rerank score to document metadata to display in the UI
            doc.metadata["rerank_score"] = float(score)
            final_docs.append(doc)
            
        context = "\n\n".join([doc.page_content for doc in final_docs])
        return final_docs, context
