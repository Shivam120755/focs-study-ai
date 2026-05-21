import json
import os

data_dir = "data"

files = os.listdir(data_dir)
print(f"Files in data/: {files}\n")

with open(os.path.join(data_dir, "focs_training_dataset.json")) as f:
    training = json.load(f)

print(f"Training Q&A pairs: {len(training)}")
print(f"Sample question: {training[0]['instruction'][:80]}...")
print(f"Sample answer:   {training[0]['response'][:80]}...")

with open(os.path.join(data_dir, "focs_raw_corpus.json")) as f:
    corpus = json.load(f)

lectures = corpus.get("lectures", {})
exams = corpus.get("exams", {})

total_chars = sum(len(v) for v in lectures.values()) + sum(len(v) for v in exams.values())

print(f"\nLectures: {len(lectures)}")
print(f"Exams/Quizzes: {len(exams)}")
print(f"Total text: {total_chars:,} characters")

from collections import Counter
topics = Counter(d.get("topic", "unknown") for d in training)
print(f"\nTopics in training data:")
for topic, count in topics.most_common():
    print(f"  {topic}: {count}")

print("\n Phase 1 complete!")