# MediAssist AI - Clinical Decision Support System

MediAssist AI is a clinical decision support application designed for healthcare professionals, researchers, and endocrinologists. It utilizes Retrieval-Augmented Generation (RAG) to synthesize medical literature from PubMed regarding Intermittent Fasting (IF), Type 2 Diabetes, obesity, and metabolic health.

The system retrieves relevant peer-reviewed studies stored in a local vector database and generates evidence-based answers with PubMed identifiers (PMID) using Groq.

## Live Demo: 

https://mediinsighthealthsolutions.streamlit.app/

## System Architecture

The application consists of three primary components:

1. PubMed Ingestion Pipeline:
   Queries the NCBI Entrez API to retrieve peer-reviewed medical abstracts and metadata, including article titles, authors, journals, and publication dates.

2. Vector Store Manager (ChromaDB):
   Embeds and stores medical abstracts as dense vectors using the all-MiniLM-L6-v2 embedding model, enabling fast semantic similarity search across clinical literature.

3. RAG Pipeline and Clinical Synthesis:
   Retrieves the most semantically relevant studies for a clinician's query and constructs a grounded prompt for the Groq LLM API. The model synthesizes key findings, clinical outcomes, and limitations while citing PMIDs.

4. Streamlit User Interface:
   Provides a web interface with a sidebar to manage the knowledge base and a query bar to ask clinical questions.

## Features

- Sidebar Knowledge Base Search: Search new PubMed keywords and ingest new articles directly into the vector database.
- Database Status Monitoring: Live display of total indexed articles in the local collection.
- Semantic Evidence Retrieval: Retrieves the top matching studies based on meaning rather than simple keyword matches.
- Evidence Grounding with Citations: Every answer references specific PMIDs, with links to the official PubMed articles.
- Preset Clinical Queries: Quick-access buttons for common research questions regarding HbA1c, insulin resistance, and fasting safety.

**Associated with:** <br>
► Codebasics