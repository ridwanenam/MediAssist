import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

class ChromaCollectionManager:
    """
    Manager module for ChromaDB vector collections.
    Handles ingestion of PubMed medical research articles and semantic similarity retrieval.
    """
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "mediassist_pubmed_if"):
        if persist_directory is None:
            self.persist_directory = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "chroma_db"
            )
        else:
            self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create the vector collection
        self.collection = self.get_or_create_collection(self.collection_name)

    def get_or_create_collection(self, collection_name: Optional[str] = None):
        """Retrieve existing collection or create a new one."""
        name = collection_name or self.collection_name
        return self.client.get_or_create_collection(
            name=name,
            metadata={"description": "PubMed articles on Intermittent Fasting for MediAssist AI"}
        )

    @staticmethod
    def _format_abstract_text(abstract_data: Any) -> str:
        """Convert abstract dictionary or string into unified clean text."""
        if isinstance(abstract_data, dict):
            sections = []
            for label, text in abstract_data.items():
                if text and text.strip():
                    if label.upper() != "SUMMARY":
                        sections.append(f"{label}: {text.strip()}")
                    else:
                        sections.append(text.strip())
            return "\n".join(sections) if sections else "No Abstract"
        elif isinstance(abstract_data, str):
            return abstract_data.strip()
        return "No Abstract"

    def ingest_articles(self, articles: List[Dict[str, Any]], batch_size: int = 50) -> int:
        """
        Ingest a list of PubMed articles into the Chroma vector collection.
        Uses upsert to prevent duplicate IDs.
        """
        if not articles:
            print("[INFO] No articles provided for ingestion.")
            return 0

        ids = []
        documents = []
        metadatas = []

        for art in articles:
            pmid = str(art.get("pmid", "")).strip()
            if not pmid:
                continue

            title = str(art.get("title", "No Title")).strip()
            journal = str(art.get("journal", "Unknown Journal")).strip()
            pub_date = str(art.get("publication_date", "Unknown Year")).strip()
            authors = str(art.get("authors", "No Authors")).strip()
            abstract_text = self._format_abstract_text(art.get("abstract", ""))

            # Build rich document text representation for semantic embeddings
            doc_content = (
                f"Title: {title}\n"
                f"Journal: {journal} ({pub_date})\n"
                f"Authors: {authors}\n"
                f"PMID: {pmid}\n\n"
                f"Abstract:\n{abstract_text}"
            )

            # Metadata must contain only primitive types (str, int, float, bool)
            meta = {
                "pmid": pmid,
                "title": title[:500],
                "journal": journal[:200],
                "publication_date": pub_date[:50],
                "authors": authors[:300]
            }

            ids.append(pmid)
            documents.append(doc_content)
            metadatas.append(meta)

        total_to_ingest = len(ids)
        print(f"[INFO] Ingesting {total_to_ingest} articles into collection '{self.collection.name}'...")

        # Ingest in batches to handle memory and payload limits gracefully
        for i in range(0, total_to_ingest, batch_size):
            b_ids = ids[i:i + batch_size]
            b_docs = documents[i:i + batch_size]
            b_metas = metadatas[i:i + batch_size]

            self.collection.upsert(
                ids=b_ids,
                documents=b_docs,
                metadatas=b_metas  # type: ignore
            )
            print(f"  -> Ingested batch {i + 1} to {min(i + batch_size, total_to_ingest)} / {total_to_ingest}")

        total_count = self.collection.count()
        print(f"[SUCCESS] Ingestion completed. Total articles in vector collection: {total_count}")
        return total_to_ingest

    def retrieve_documents(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store with semantic search and return the top-N matching articles.
        """
        if self.collection.count() == 0:
            print("[WARNING] Vector collection is empty. Please ingest articles first.")
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count())
        )

        formatted_results: List[Dict[str, Any]] = []
        raw_docs = results.get("documents")
        if raw_docs and len(raw_docs) > 0 and raw_docs[0] is not None:
            docs = raw_docs[0]
            raw_metas = results.get("metadatas")
            metas = raw_metas[0] if raw_metas and len(raw_metas) > 0 and raw_metas[0] is not None else [{}] * len(docs)
            raw_ids = results.get("ids")
            ids = raw_ids[0] if raw_ids and len(raw_ids) > 0 and raw_ids[0] is not None else [""] * len(docs)
            raw_dists = results.get("distances")
            distances = raw_dists[0] if raw_dists and len(raw_dists) > 0 and raw_dists[0] is not None else [0.0] * len(docs)

            for doc, meta, doc_id, dist in zip(docs, metas, ids, distances):
                meta_dict = meta if isinstance(meta, dict) else {}
                formatted_results.append({
                    "pmid": doc_id,
                    "title": meta_dict.get("title", ""),
                    "journal": meta_dict.get("journal", ""),
                    "publication_date": meta_dict.get("publication_date", ""),
                    "authors": meta_dict.get("authors", ""),
                    "document": doc,
                    "similarity_distance": dist
                })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Return collection metadata and document count."""
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "persist_directory": self.persist_directory
        }
