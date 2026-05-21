import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

print('Building database with cleaned textbook...')

print('[1/4] Loading materials...')
with open('data/focs_raw_corpus.json') as f:
    corpus = json.load(f)
with open('data/focs_training_dataset.json') as f:
    training_data = json.load(f)
with open('data/textbook_cleaned.json') as f:
    textbook = json.load(f)
print('  Lectures:', len(corpus.get('lectures', {})))
print('  Exams:', len(corpus.get('exams', {})))
print('  QA pairs:', len(training_data))
print('  Textbook pages:', len(textbook))

print('[2/4] Chunking...')
documents = []
sources = []

def chunk_text(text, size=400, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        c = ' '.join(words[i:i+size])
        if len(c.strip()) > 50:
            chunks.append(c.strip())
    return chunks

for name, text in corpus.get('lectures', {}).items():
    for chunk in chunk_text(text):
        documents.append(chunk)
        sources.append(name)

for name, text in corpus.get('exams', {}).items():
    for chunk in chunk_text(text):
        documents.append(chunk)
        sources.append(name)

for name, text in textbook.items():
    for chunk in chunk_text(text):
        documents.append(chunk)
        sources.append(name)

for item in training_data:
    documents.append('Q: ' + item['instruction'] + ' A: ' + item['response'])
    sources.append(item.get('source', 'training'))

print(f'  Total chunks: {len(documents)}')

print('[3/4] Creating embeddings (few minutes)...')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=64)
print(f'  Shape: {embeddings.shape}')

print('[4/4] Saving...')
db = {'documents': documents, 'sources': sources, 'embeddings': embeddings}
with open('data/vectordb.pkl', 'wb') as f:
    pickle.dump(db, f)
print('  Saved!')

print()
print('Testing...')
def search(query, top_k=3):
    q_emb = embedder.encode([query])
    scores = np.dot(embeddings, q_emb.T).squeeze()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [(documents[i], sources[i], scores[i]) for i in top_idx]

for q in ['What is the Handshaking Theorem?', 'Explain strong induction', 'What is a Turing machine?']:
    print(f"Query: '{q}'")
    for j, (doc, src, score) in enumerate(search(q)):
        print(f'  [{src}] (score:{score:.3f}): {doc[:80]}...')
    print()
print('Done! Database now includes the textbook.')
