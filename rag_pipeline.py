import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq

# Ensure current script directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from chroma_manager import ChromaCollectionManager  # type: ignore[import-not-found]

class MedicalRAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) Pipeline for MediAssist AI.
    Combines PubMed vector retrieval from ChromaDB with Groq LLM generation.
    """

    def __init__(
        self,
        collection_name: str = "mediassist_pubmed_if",
        persist_directory: Optional[str] = None,
        preferred_model: Optional[str] = None,
        temperature: float = 0.2
    ):
        # 1. Load Environment Variables (.env)
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        load_dotenv(dotenv_path=env_path)

        # Check Streamlit Cloud secrets first, then environment variable
        self.api_key = None
        try:
            import streamlit as st  # type: ignore
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                self.api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

        if not self.api_key:
            self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key or self.api_key.strip() == "" or self.api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is not configured. Please set your valid Groq API key in MediAssist-AI/.env or Streamlit Secrets."
            )

        # 2. Initialize Groq Client
        self.client = Groq(api_key=self.api_key)
        self.temperature = temperature

        # 3. Resolve Active LLM Model
        self.model = self._resolve_model(preferred_model)

        # 4. Connect to ChromaDB Vector Store
        if persist_directory is None:
            persist_directory = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "chroma_db"
            )
        self.chroma_manager = ChromaCollectionManager(
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        doc_count = self.chroma_manager.collection.count()
        print(f"[INIT] MedicalRAGPipeline ready.")
        print(f"       Model: {self.model}")
        print(f"       Vector Collection: '{collection_name}' ({doc_count} indexed articles)")

    def _resolve_model(self, preferred: Optional[str]) -> str:
        """
        Determines the most capable active model available on the user's Groq account.
        Checks preferred model first, then known high-performance models.
        """
        try:
            available = [m.id for m in self.client.models.list().data]
        except Exception as e:
            print(f"[WARNING] Could not query Groq models list ({e}). Defaulting to preferred or llama model.")
            return preferred or "llama-3.3-70b-versatile"

        candidates = []
        if preferred:
            candidates.append(preferred)

        # Priority order of models: LLaMA 3 variants first, then high-parameter models on Groq
        candidates.extend([
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b"
        ])

        for c in candidates:
            if c in available:
                return c

        # If none matched, return first available text chat model
        if available:
            chat_models = [m for m in available if not ("whisper" in m or "guard" in m)]
            if chat_models:
                return chat_models[0]
            return available[0]

        return "llama-3.3-70b-versatile"

    def retrieve_context(self, query: str, n_results: int = 4) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for the top-N semantically relevant PubMed articles.
        """
        return self.chroma_manager.retrieve_documents(query_text=query, n_results=n_results)

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved PubMed documents into a structured reference text for the prompt.
        """
        if not retrieved_docs:
            return "No relevant medical studies found in the local vector database."

        formatted_blocks = []
        for idx, doc in enumerate(retrieved_docs, start=1):
            pmid = doc.get("pmid", "N/A")
            title = doc.get("title", "No Title")
            journal = doc.get("journal", "Unknown Journal")
            pub_date = doc.get("publication_date", "N/A")
            authors = doc.get("authors", "Unknown Authors")
            doc_text = doc.get("document", "")

            block = (
                f"[Document {idx}]\n"
                f"PMID: {pmid}\n"
                f"Title: {title}\n"
                f"Journal: {journal} ({pub_date})\n"
                f"Authors: {authors}\n"
                f"Content:\n{doc_text}\n"
            )
            formatted_blocks.append(block)

        return "\n" + ("-" * 50) + "\n" + ("\n" + ("-" * 50) + "\n").join(formatted_blocks)

    def build_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        Creates a system and user prompt with Context and Query variables,
        dynamically enforcing the response language based on user query.
        """
        id_words = [
            "apakah", "bagaimana", "mengapa", "kenapa", "apa", "berapa", "dan", "yang",
            "pada", "untuk", "dengan", "atau", "adalah", "efek", "pasien", "puasa", "bisa", "aman"
        ]
        q_lower = f" {query.lower().strip()} "
        is_indonesian = any(f" {w} " in q_lower for w in id_words)

        if is_indonesian:
            lang_instruction = (
                "INSTRUKSI BAHASA WAJIB:\n"
                "Pengguna mengajukan pertanyaan dalam BAHASA INDONESIA. "
                "Kamu WAJIB menyusun seluruh jawaban, analisis, dan ringkasan klinis dalam BAHASA INDONESIA yang profesional dan terstruktur.\n"
                "Gunakan struktur bagian berikut:\n"
                "- Temuan Klinis Utama\n"
                "- Bukti Ilmiah & Mekanisme\n"
                "- Pertimbangan Praktis / Keterbatasan\n"
                "Tetap pertahankan istilah medis (seperti HbA1c, HOMA-IR, Intermittent Fasting) dan sitasi [PMID: xxxxx] dengan tepat."
            )
            user_trailer = (
                "Jawablah pertanyaan di atas secara komprehensif, berbasis bukti, dan SELURUHNYA DALAM BAHASA INDONESIA "
                "menggunakan dokumen konteks yang tersedia."
            )
        else:
            lang_instruction = (
                "Language Guideline:\n"
                "Respond in clear, professional English. Organize into clear sections: "
                "Key Clinical Findings, Scientific Evidence & Mechanisms, and Practical Considerations / Limitations."
            )
            user_trailer = (
                "Please provide a comprehensive, evidence-based clinical answer using the context provided above."
            )

        system_instruction = (
            "You are MediAssist AI, an advanced clinical intelligence assistant developed for "
            "MediInsight Health Solutions. Your purpose is to assist healthcare practitioners, "
            "endocrinologists, and clinicians by synthesizing medical literature regarding "
            "Intermittent Fasting (IF), diabetes, obesity, and metabolic health.\n\n"
            "Strict Guidelines:\n"
            "1. Ground your synthesis strictly in the provided Context documents.\n"
            "2. Cite PMIDs and study titles directly when presenting clinical findings (e.g. '[PMID: 12345678]').\n"
            "3. If the context does not contain sufficient clinical evidence to answer a point, explicitly state: "
            "'Based on the retrieved studies, insufficient data is available.' (atau 'Berdasarkan studi yang ditemukan, data belum mencukupi.'). "
            "Do not speculate or invent medical facts.\n"
            f"4. {lang_instruction}"
        )

        user_content = (
            f"Context:\n{context}\n\n"
            f"Query:\n{query}\n\n"
            f"{user_trailer}"
        )

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    def generate_answer(self, query: str, n_results: int = 4) -> Dict[str, Any]:
        """
        Executes the end-to-end RAG pipeline:
        1. Query vector store for relevant PubMed documents
        2. Build prompt with Context and Query
        3. Call Groq LLM API to generate a clinical response
        """
        # Step 1: Retrieve context documents from ChromaDB
        retrieved_docs = self.retrieve_context(query=query, n_results=n_results)

        # Step 2: Format context and generate prompt
        formatted_context = self.format_context(retrieved_docs)
        messages = self.build_prompt(query=query, context=formatted_context)

        # Step 3: Call Groq API
        try:
            messages_payload: Any = messages
            response: Any = self.client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=messages_payload,
                temperature=self.temperature,
                max_tokens=1024
            )
            answer_text = response.choices[0].message.content or "No response generated."
        except Exception as e:
            answer_text = f"Error during generation via Groq API: {str(e)}"

        return {
            "query": query,
            "answer": answer_text,
            "context_docs": retrieved_docs,
            "model_used": self.model,
            "num_retrieved": len(retrieved_docs)
        }

def run_test():
    """
    Test script to verify the RAG pipeline execution on a clinical query.
    """
    print("=" * 70)
    print(" MediAssist AI - RAG Pipeline Test (Email 2)")
    print("=" * 70)

    pipeline = MedicalRAGPipeline()

    sample_query = (
        "What are the clinical effects of intermittent fasting on insulin resistance, "
        "glycemic control, and weight loss in patients with type 2 diabetes?"
    )

    print(f"\n[QUERY] {sample_query}")
    print("\nRetrieving evidence from ChromaDB and generating response with Groq...")

    result = pipeline.generate_answer(query=sample_query, n_results=4)

    print("\n" + "=" * 70)
    print(f" RETRIEVED SOURCES ({result['num_retrieved']} articles from ChromaDB)")
    print("=" * 70)
    for idx, doc in enumerate(result["context_docs"], start=1):
        print(f" [{idx}] PMID: {doc['pmid']} | {doc['title']}")
        print(f"     Journal: {doc['journal']} ({doc['publication_date']})")
        print(f"     Distance: {doc['similarity_distance']:.4f}\n")

    print("=" * 70)
    print(f" GENERATED CLINICAL RESPONSE (Model: {result['model_used']})")
    print("=" * 70)
    print(result["answer"])
    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_test()
