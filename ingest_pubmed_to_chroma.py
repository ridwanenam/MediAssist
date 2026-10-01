import sys
import os
import time

# Ensure current script directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pubmed import PubMedRetriever  # type: ignore[import-not-found]
from chroma_manager import ChromaCollectionManager  # type: ignore[import-not-found]

def run_ingestion_pipeline(search_term: str = "intermittent fasting obesity diabetes", max_articles: int = 150):
    """
    Executes the complete Email 1 data ingestion workflow:
    1. Search PubMed for PMIDs (capped at max 300 to respect rate limits)
    2. Fetch article abstracts and metadata
    3. Ingest and persist into ChromaDB vector store
    4. Perform a sample semantic retrieval query to verify the collection
    """
    print("=" * 70)
    print(" MediAssist AI - PubMed Data Ingestion & ChromaDB Setup (Email 1)")
    print("=" * 70)

    # 1. Retrieve PubMed Article IDs (PMIDs)
    capped_max = min(max_articles, 300)
    print(f"\n[STEP 1] Searching PubMed articles for term: '{search_term}'...")
    print(f"         Article limit: {capped_max} (Strictly <= 300 to respect NCBI rate limits)")
    
    t0 = time.time()
    pmids = PubMedRetriever.search_pubmed_articles(search_term=search_term, max_results=capped_max)
    t_search = time.time() - t0
    
    print(f"[SUCCESS] Retrieved {len(pmids)} PMIDs in {t_search:.2f} seconds.")
    if not pmids:
        print("[ERROR] No PMIDs retrieved. Exiting.")
        return

    # 2. Iterate Through PMIDs and Retrieve Abstracts & Metadata
    print(f"\n[STEP 2] Fetching abstracts and metadata for {len(pmids)} PMIDs from PubMed...")
    t1 = time.time()
    articles = PubMedRetriever.fetch_pubmed_abstracts(pmids)
    t_fetch = time.time() - t1
    print(f"[SUCCESS] Successfully fetched {len(articles)} article records in {t_fetch:.2f} seconds.")

    # 3. Create and Ingest into Chroma Collection
    print(f"\n[STEP 3] Initializing ChromaDB vector store and ingesting articles...")
    manager = ChromaCollectionManager(
        persist_directory="./data/chroma_db",
        collection_name="mediassist_pubmed_if"
    )

    t2 = time.time()
    ingested_count = manager.ingest_articles(articles, batch_size=50)
    t_ingest = time.time() - t2
    print(f"[SUCCESS] Ingested {ingested_count} articles into ChromaDB in {t_ingest:.2f} seconds.")

    # 4. Verify Retrieval with Sample Medical Query
    print("\n" + "=" * 70)
    print(" [STEP 4] Verifying Semantic Retrieval from ChromaDB")
    print("=" * 70)
    
    sample_query = "What are the effects of intermittent fasting on insulin resistance and weight loss?"
    print(f"Sample Query: '{sample_query}'")
    
    results = manager.retrieve_documents(query_text=sample_query, n_results=3)
    
    print(f"\nTop {len(results)} Relevant PubMed Articles Retrieved:")
    for idx, r in enumerate(results, 1):
        print(f"\n--- Result #{idx} (Similarity Distance: {r['similarity_distance']:.4f}) ---")
        print(f"PMID: {r['pmid']}")
        print(f"Title: {r['title']}")
        print(f"Journal: {r['journal']} ({r['publication_date']})")
        print(f"Authors: {r['authors']}")
        # First 200 chars of document text
        snippet = r['document'].split("Abstract:\n")[-1][:250].strip()
        print(f"Abstract Snippet: {snippet}...")

    stats = manager.get_stats()
    print("\n" + "=" * 70)
    print(f" INGESTION PIPELINE COMPLETE")
    print(f" Collection: {stats['collection_name']} | Total Documents: {stats['document_count']}")
    print("=" * 70)

if __name__ == "__main__":
    # Default ingestion of 150 relevant articles (well within the <=300 limit)
    run_ingestion_pipeline(search_term="intermittent fasting diabetes obesity", max_articles=150)
