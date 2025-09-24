# Python and AI Power-up Program - RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python that allows users to ask questions about AI and machine learning topics based on PDF transcripts. The application uses Qdrant for vector storage, FastEmbed for embeddings, Google's Gemini for language generation, and Streamlit for the user interface.

## 📋 Prerequisites

- Python 3.9 or higher (recommended: 3.10+)
- Qdrant Cloud account or self-hosted Qdrant instance
- Google AI API key for Gemini
- Opik account for observability (optional)

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd python-bot
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

| Variable            | Description                    | Required | Example                                                   |
| ------------------- | ------------------------------ | -------- | --------------------------------------------------------- |
| `QDRANT_URL`        | Your Qdrant instance URL       | Yes      | `https://xyz-example.us-east4-0.gcp.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY`    | Qdrant API key                 | Yes      | `your-qdrant-api-key`                                     |
| `QDRANT_COLLECTION` | Collection name for vectors    | No       | `transcripts-v1` (default)                                |
| `GOOGLE_API_KEY`    | Google AI API key for Gemini   | Yes      | `your-google-api-key`                                     |
| `OPIK_API_KEY`      | Opik API key for observability | No       | `your-opik-api-key`                                       |
| `OPIK_WORKSPACE`    | Opik workspace name            | No       | `default`                                                 |

### Model Configuration

The application uses the following models and settings (configurable in `app.py`):

- **Dense Embeddings**: `jinaai/jina-embeddings-v2-base-en` (768 dimensions)
- **Sparse Embeddings**: `Qdrant/BM25`
- **LLM**: `gemini-2.5-pro`
- **Retrieval Settings**:
  - `TOP_K = 4` (number of contexts to retrieve)
  - `MMR_DIVERSITY = 0.3` (diversity factor for MMR reranking)

## 🚀 Usage

### 1. Data Ingestion

First, add your PDF files to the `data/` directory, then run the ingestion script:

```bash
cd src
python ingest.py
```

This will:

- Load all PDFs from the `data/` directory
- Split documents into chunks (2048 characters)
- Generate dense and sparse embeddings
- Store vectors in Qdrant collection

### 2. Running the Application

Start the Streamlit app:

```bash
cd src
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### 3. Using the Chatbot

1. **Ask Questions**: Enter AI/ML related questions in the text input
2. **Filter by Source**: Use the sidebar to filter results by specific PDF files
3. **View Responses**: Get contextual answers with source attribution
4. **Review Context**: See the retrieved text chunks that informed the answer

## 🔧 Customization

### Adding New PDFs

1. Place PDF files in the `data/` directory
2. Update the `pdfs` list in `ingest.py` with the new file paths
3. Re-run the ingestion script: `python ingest.py`

### Modifying Retrieval Settings

Edit the constants in `app.py`:

```python
TOP_K = 4              # Number of contexts to retrieve
MMR_DIVERSITY = 0.3    # Diversity factor (0.0 = no diversity, 1.0 = max diversity)
```

### Changing the System Prompt

Modify the `system_prompt` in the `generate_res` function in `app.py` to customize the AI's behavior.

### Using Different Models

Update the model configurations in `app.py`:

```python
# For different embedding models
DENSE_MODEL = TextEmbedding(model_name="your-preferred-model")

# For different LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # or other available models
    temperature=0.0,
)
```

## 📚 Dependencies

Key dependencies include:

- **Streamlit**: Web application framework
- **Qdrant Client**: Vector database client
- **LangChain**: LLM framework and utilities
- **FastEmbed**: Fast embedding generation
- **Google GenerativeAI**: Gemini LLM integration
- **PyPDFium2**: PDF processing
- **Opik**: Observability and tracing

See `requirements.txt` for complete list with versions.
