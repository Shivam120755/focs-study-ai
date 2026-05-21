import json
import numpy as np
import pickle
import gradio as gr
from sentence_transformers import SentenceTransformer

print('Loading vector database...')
with open('data/vectordb.pkl', 'rb') as f:
    db = pickle.load(f)

documents = db['documents']
sources = db['sources']
embeddings = db['embeddings']

print('Loading embedding model...')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print(f'Ready! {len(documents)} documents loaded.')

def search_only(question):
    if not question.strip():
        return 'Please enter a question.'
    q_emb = embedder.encode([question])
    scores = np.dot(embeddings, q_emb.T).squeeze()
    top_idx = np.argsort(scores)[-5:][::-1]
    results = ''
    for rank, idx in enumerate(top_idx):
        src = sources[idx]
        score = scores[idx]
        text = documents[idx]
        results += f'--- Result {rank+1} [Source: {src}] (similarity: {score:.3f}) ---\n'
        results += text + '\n\n'
    return results

with gr.Blocks(title='FOCS Study AI - Search Only') as app:
    gr.Markdown('# System A: Search Only (No LLM)')
    gr.Markdown('Type a FOCS question. You will see the raw text chunks from course materials. No AI summarization.')
    question = gr.Textbox(label='Your Question', placeholder='e.g. What is the Handshaking Theorem?')
    submit = gr.Button('Search', variant='primary')
    output = gr.Textbox(label='Raw Search Results', lines=20)
    submit.click(fn=search_only, inputs=question, outputs=output)
    question.submit(fn=search_only, inputs=question, outputs=output)

app.launch()
