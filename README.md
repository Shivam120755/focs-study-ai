# FOCS Study AI

An AI-powered study assistant for **RPI's CSCI 2200 — Foundations of Computer Science (FOCS)**, built with Retrieval-Augmented Generation (RAG). It answers course questions using the actual textbook, lecture slides, and past exams — running entirely on a local GPU with no external API calls.

## Demo

The app provides a side-by-side comparison that demonstrates the value of the LLM in a RAG pipeline:
- **AI Answer (RAG):** A clear, synthesized answer grounded in course material
- **Retrieved Chunks (Search Only):** The raw text chunks the search returned, before the LLM processed them

## What It Does

Ask any FOCS question and get an answer based on the specific way the course teaches it — Professor Magdon-Ismail's notation, real exam-style problems, and explanations drawn from the actual course textbook. Unlike a general chatbot, every answer is grounded in retrieved course content.

## How It Works

This project uses **RAG (Retrieval-Augmented Generation)**:

```
User Question
      |
      v
[1] Embed question into a 384-dim vector
      |
      v
[2] Search ~5,000 course-material chunks by vector similarity
      |
      v
[3] Retrieve top 3 most relevant chunks
      |
      v
[4] Insert chunks + question into the LLM prompt
      |
      v
[5] LLM (Qwen2.5-3B) generates a course-specific answer
```

**Retrieval** finds the relevant course content. **Generation** synthesizes it into a coherent answer. The key insight: the LLM provides general language ability, while the retrieved course material provides the specific knowledge.

## Architecture

| Component | Technology | Role |
|-----------|-----------|------|
| Language Model | Qwen2.5-3B-Instruct | Generates answers (runs locally on GPU in float16) |
| Embeddings | all-MiniLM-L6-v2 | Converts text to 384-dim vectors (runs on CPU) |
| Vector Search | NumPy dot-product | Semantic similarity search over course chunks |
| Interface | Gradio | Web UI |

## Data Sources

- Official course textbook (459 pages)
- 29 lecture slide sets
- 26 exams and quizzes (2014–2020)
- 92 curated question-answer pairs

All materials are chunked into ~400-word overlapping segments and embedded into a searchable vector database.

## Setup

### Prerequisites
- Python 3.12
- NVIDIA GPU with 6GB+ VRAM
- The course materials (not included in this repo)

### Install
```bash
git clone https://github.com/YOUR_USERNAME/focs-study-ai.git
cd focs-study-ai
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentence-transformers gradio numpy
```

### Build the vector database
Place your course material JSON files in `data/`, then:
```bash
python scripts/build_vectordb.py
```

### Run the app
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

## Project Structure

```
focs-study-ai/
├── app.py                    # Main Gradio application (RAG pipeline + UI)
├── scripts/
│   └── build_vectordb.py     # Builds the vector database from course materials
├── data/                     # Course materials and vector DB (gitignored)
└── README.md
```

## Key Engineering Decisions

**Why RAG instead of fine-tuning?** Fine-tuning teaches style and patterns but doesn't reliably encode facts. RAG stores knowledge externally and retrieves it at query time, which means answers stay accurate and the knowledge base can be updated without retraining.

**Why a local model?** Running Qwen2.5-3B locally means no API costs, no data leaving the machine, and full control over the pipeline. The 3B model fits comfortably on an 8GB laptop GPU in float16.

**Why semantic search over keyword search?** Embeddings match by meaning, so a question like "How many edges in a friendship network?" correctly retrieves content about the Handshaking Theorem even without exact keyword overlap.

**Data cleaning:** The textbook was a scanned PDF with OCR artifacts. A cleaning pipeline fixes systematic errors (like spaced-out capital letters and common character substitutions) before the text enters the knowledge base.

## Possible Improvements

- Add a reranking step (cross-encoder) after initial retrieval for higher precision
- Implement conversational memory for multi-turn follow-up questions
- Build an evaluation benchmark of held-out exam questions to quantify accuracy
- Use multimodal extraction to capture diagrams and figures from the textbook

## Tech Stack

Python · PyTorch · Transformers · Sentence-Transformers · Gradio · NumPy

---

*Built as a learning project to understand RAG systems. CSCI 2200 Foundations of Computer Science, Rensselaer Polytechnic Institute.*
