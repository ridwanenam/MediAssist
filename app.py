import os
import sys
import time
from typing import List, Dict, Any
import streamlit as st

# Ensure current script directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pubmed import PubMedRetriever  # type: ignore[import-not-found]
from chroma_manager import ChromaCollectionManager  # type: ignore[import-not-found]
from rag_pipeline import MedicalRAGPipeline  # type: ignore[import-not-found]

# Page configuration
st.set_page_config(
    page_title="MediAssist AI - Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean CSS without decorative symbols
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .source-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .source-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .source-meta {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 6px;
    }
    .source-abstract {
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_pipeline() -> MedicalRAGPipeline:
    """Initialize and cache the Medical RAG Pipeline instance."""
    return MedicalRAGPipeline()

@st.cache_resource
def get_chroma_manager() -> ChromaCollectionManager:
    """Initialize and cache the ChromaCollectionManager instance."""
    return ChromaCollectionManager()

def main():
    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sample_query_trigger" not in st.session_state:
        st.session_state.sample_query_trigger = None

    # Load cached services
    try:
        pipeline = get_rag_pipeline()
        chroma_manager = get_chroma_manager()
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.info("Please verify that your GROQ_API_KEY is configured in MediAssist-AI/.env.")
        st.stop()

    current_doc_count = chroma_manager.collection.count()

    # ---------------------------------------------------------
    # SIDEBAR: PubMed Article Search & Knowledge Ingestion
    # ---------------------------------------------------------
    with st.sidebar:
        st.subheader("Knowledge Base Management")
        st.caption("Manage local PubMed vector collection for evidence retrieval.")

        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600;">Indexed Articles</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #0284c7;">{current_doc_count} Documents</div>
            <div style="font-size: 0.75rem; color: #64748b;">Collection: {chroma_manager.collection.name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.write("**Search and Ingest New Articles**")

        search_query = st.text_input(
            "PubMed Search Term",
            value="intermittent fasting diabetes obesity",
            help="Enter medical topics, conditions, or clinical keywords."
        )

        max_articles = st.slider(
            "Number of Articles to Fetch",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
            help="Number of PubMed articles to download and vectorize into ChromaDB."
        )

        ingest_button = st.button("Ingest Articles into Vector Store", use_container_width=True)

        if ingest_button:
            if not search_query.strip():
                st.warning("Please provide a search term.")
            else:
                with st.status("Fetching and ingesting PubMed articles...", expanded=True) as status_box:
                    try:
                        retriever = PubMedRetriever()
                        st.write("Searching PubMed ID records via NCBI Entrez API...")
                        pmids = retriever.search_articles(query=search_query.strip(), max_results=max_articles)

                        if not pmids:
                            st.warning(f"No PubMed articles found for query: '{search_query}'.")
                            status_box.update(label="No articles found", state="error")
                        else:
                            st.write(f"Retrieved {len(pmids)} PMIDs. Fetching complete abstracts and metadata...")
                            articles = retriever.fetch_article_details(pmids)

                            st.write(f"Embedding and storing {len(articles)} articles into ChromaDB...")
                            chroma_manager.add_articles_to_collection(articles)

                            new_count = chroma_manager.collection.count()
                            status_box.update(label=f"Ingestion Complete: {len(articles)} articles processed.", state="complete")
                            st.success(f"Vector store updated. Total articles now indexed: {new_count}")
                            time.sleep(1)
                            st.rerun()
                    except Exception as ex:
                        status_box.update(label="Ingestion failed", state="error")
                        st.error(f"Error during ingestion: {str(ex)}")

        st.markdown("---")
        st.write("**System Configuration**")
        st.text(f"LLM Engine: {pipeline.model}")
        st.text("Embedding: all-MiniLM-L6-v2 (384-d)")
        st.text("Vector Store: ChromaDB (Local)")

    # ---------------------------------------------------------
    # MAIN AREA: Query Bar & Clinical Decision Support
    # ---------------------------------------------------------
    st.markdown('<div class="main-header">MediAssist AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Clinical Decision Support for Intermittent Fasting and Metabolic Disorders</div>',
        unsafe_allow_html=True
    )

    if current_doc_count == 0:
        st.info("The vector store is currently empty. Use the sidebar on the left to search PubMed and click 'Ingest Articles into Vector Store' to populate your clinical knowledge base.")


    # Preset Quick Clinical Questions
    st.write("**Quick Clinical Question Examples:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Effects on HbA1c and Insulin Resistance", use_container_width=True):
            st.session_state.sample_query_trigger = (
                "What are the documented clinical effects of intermittent fasting on insulin resistance, "
                "glycemic control, and HbA1c in patients with type 2 diabetes?"
            )
    with col2:
        if st.button("Cardiometabolic Risks and Safety", use_container_width=True):
            st.session_state.sample_query_trigger = (
                "What are the potential cardiometabolic risks, adverse events, or contraindications "
                "associated with intermittent fasting?"
            )
    with col3:
        if st.button("Time-Restricted Eating vs Calorie Restriction", use_container_width=True):
            st.session_state.sample_query_trigger = (
                "How does time-restricted eating compare with continuous calorie restriction "
                "regarding weight loss and body composition in obesity?"
            )

    # Render previous conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander(f"Referenced PubMed Literature ({len(message['sources'])} Studies)"):
                    for s_idx, src in enumerate(message["sources"], start=1):
                        pmid = src.get("pmid", "N/A")
                        title = src.get("title", "No Title")
                        journal = src.get("journal", "Unknown Journal")
                        pub_date = src.get("publication_date", "N/A")
                        authors = src.get("authors", "Unknown Authors")
                        dist = src.get("similarity_distance", 0.0)

                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">{s_idx}. <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">{title}</a></div>
                            <div class="source-meta">PMID: <strong>{pmid}</strong> | Journal: {journal} ({pub_date}) | Authors: {authors} | Distance: {dist:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Handle Input Query
    user_query = st.chat_input("Enter your clinical or research question...")

    # Check if a preset button was clicked
    if st.session_state.sample_query_trigger:
        user_query = st.session_state.sample_query_trigger
        st.session_state.sample_query_trigger = None

    if user_query:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response using RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant PubMed abstracts and synthesizing clinical evidence..."):
                result = pipeline.generate_answer(query=user_query, n_results=4)
                answer_text = result["answer"]
                retrieved_sources = result["context_docs"]

                st.markdown(answer_text)

                if retrieved_sources:
                    with st.expander(f"Referenced PubMed Literature ({len(retrieved_sources)} Studies)"):
                        for s_idx, src in enumerate(retrieved_sources, start=1):
                            pmid = src.get("pmid", "N/A")
                            title = src.get("title", "No Title")
                            journal = src.get("journal", "Unknown Journal")
                            pub_date = src.get("publication_date", "N/A")
                            authors = src.get("authors", "Unknown Authors")
                            dist = src.get("similarity_distance", 0.0)

                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-title">{s_idx}. <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">{title}</a></div>
                                <div class="source-meta">PMID: <strong>{pmid}</strong> | Journal: {journal} ({pub_date}) | Authors: {authors} | Distance: {dist:.4f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": retrieved_sources
                })

if __name__ == "__main__":
    main()
