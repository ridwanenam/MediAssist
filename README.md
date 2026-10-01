# MediAssist AI - Clinical Decision Support System

MediAssist AI is a clinical decision support application designed for healthcare professionals, researchers, and endocrinologists. It utilizes Retrieval-Augmented Generation (RAG) to synthesize medical literature from PubMed regarding Intermittent Fasting (IF), Type 2 Diabetes, obesity, and metabolic health.

The system retrieves relevant peer-reviewed studies stored in a local vector database and generates evidence-based answers with PubMed identifiers (PMID) using Groq.

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

## Project Structure

- app.py: Streamlit web application providing the user interface.
- rag_pipeline.py: RAG pipeline connecting ChromaDB retrieval with Groq generation.
- chroma_manager.py: Vector database manager handling collection indexing and similarity search.
- pubmed.py: Client module for searching and fetching articles from the NCBI Entrez PubMed API.
- ingest_pubmed_to_chroma.py: Automated pipeline script to batch ingest PubMed articles into ChromaDB.
- requirements.txt: Python package dependencies.
- .env.example: Example configuration file for environment variables.
- .gitignore: Git ignore configuration to prevent committing credentials and local databases.

## Installation and Setup

### 1. Clone or Open the Repository

Navigate to the project directory:

```bash
cd MediAssist-AI
```

### 2. Set Up a Virtual Environment

Create and activate a Python virtual environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a .env file in the MediAssist-AI directory based on .env.example:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace `your_groq_api_key_here` with your API key from https://console.groq.com.

## Running the Application

### Option A: Ingest Initial Data via Command Line

To populate the local ChromaDB vector store with articles:

```bash
python ingest_pubmed_to_chroma.py
```

This will search PubMed for articles related to intermittent fasting, diabetes, and obesity, and index them into `data/chroma_db/`.

### Option B: Test the RAG Pipeline via Command Line

To test retrieval and response generation in the terminal:

```bash
python rag_pipeline.py
```

### Option C: Launch the Streamlit Web Application

To start the interactive web interface:

```bash
streamlit run app.py
```

Once running, access the application in your browser at `http://localhost:8501`.

## Features

- Sidebar Knowledge Base Search: Search new PubMed keywords and ingest new articles directly into the vector database.
- Database Status Monitoring: Live display of total indexed articles in the local collection.
- Semantic Evidence Retrieval: Retrieves the top matching studies based on meaning rather than simple keyword matches.
- Evidence Grounding with Citations: Every answer references specific PMIDs, with links to the official PubMed articles.
- Preset Clinical Queries: Quick-access buttons for common research questions regarding HbA1c, insulin resistance, and fasting safety.
