import json
import numpy as np
import pickle
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading FOCS Study AI...")
print("  Loading vector database...")
with open("data/vectordb.pkl", "rb") as f:
    db = pickle.load(f)
documents = db["documents"]
sources = db["sources"]
embeddings = db["embeddings"]

print("  Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("  Loading LLM...")
llm_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype="float16", device_map="auto")
print("Ready!")

def search(query, top_k=5):
    q_emb = embedder.encode([query])
    scores = np.dot(embeddings, q_emb.T).squeeze()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [{"text": documents[i], "source": sources[i], "score": float(scores[i])} for i in top_idx]

def generate(messages, max_tokens=500):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3000).to(llm.device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True, repetition_penalty=1.15)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def ask_question(question):
    if not question.strip():
        return "Please enter a question.", ""
    results = search(question, top_k=3)
    context = "\n\n".join([r["text"] for r in results])
    source_list = ", ".join(set([r["source"] for r in results]))
    raw_chunks = ""
    for i, r in enumerate(results):
        src = r["source"]
        sc = r["score"]
        txt = r["text"][:300]
        raw_chunks += "Result " + str(i+1) + " [" + src + "] (score: " + str(round(sc,3)) + "):\n" + txt + "...\n\n"
    messages = [
        {"role": "system", "content": "You are a study assistant for FOCS (Foundations of Computer Science) at RPI. Answer using the provided course material. Be clear and well-organized. Use Markdown formatting: ## for headers, ** for bold key terms, bullet points for lists. For math, use LaTeX with \\( \\) for inline math and \\[ \\] for display equations. Show your work step by step for problems."},
        {"role": "user", "content": f"Course material:\n{context}\n\nQuestion: {question}"}
    ]
    answer = generate(messages)
    answer += f"\n\n---\n*Sources: {source_list}*"
    return answer, raw_chunks

latex_delims = [
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$$", "right": "$$", "display": True},
]

hide_footer = "footer {visibility: hidden}"

with gr.Blocks(title="FOCS Study AI", theme=gr.themes.Soft(primary_hue="teal"), css=hide_footer) as app:
    gr.Markdown("# FOCS Study AI")
    gr.Markdown("*AI study tool for RPI CSCI 2200 - Foundations of Computer Science*")

    with gr.Tab("Ask a Question"):
        question = gr.Textbox(label="Your Question", placeholder="e.g. What is the Handshaking Theorem? How do I prove by induction?", lines=2)
        ask_btn = gr.Button("Ask", variant="primary", size="lg")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### AI Answer (RAG)")
                answer_out = gr.Markdown(latex_delimiters=latex_delims)
            with gr.Column(scale=1):
                raw_out = gr.Textbox(label="Retrieved Chunks (Search Only - No RAG)", lines=15)
        ask_btn.click(fn=ask_question, inputs=question, outputs=[answer_out, raw_out])
        question.submit(fn=ask_question, inputs=question, outputs=[answer_out, raw_out])

    with gr.Tab("How It Works"):
        gr.Markdown("""
### System Architecture

This tool uses **RAG (Retrieval-Augmented Generation)**:

**1. Search:** Your question is embedded into a vector and compared against thousands of chunks of course material (textbook, lectures, exams) to find the most relevant content.

**2. Augment:** The retrieved material is added to the prompt as context.

**3. Generate:** A 3B parameter LLM (Qwen2.5-3B) reads the context and generates a course-specific answer.

### Data Sources
- Official course textbook (459 pages)
- 29 lecture slide sets
- 26 exams and quizzes (2014-2020)
- 92 curated Q&A pairs

### Tech Stack
- **LLM:** Qwen2.5-3B-Instruct (local GPU)
- **Embeddings:** all-MiniLM-L6-v2
- **Search:** NumPy vector similarity
- **Interface:** Gradio
        """)

app.launch()
