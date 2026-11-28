import pandas as pd
import numpy as np
import math
import re
import unicodedata
from collections import Counter

def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)

seed_all(42)

def norm(s):
    s = unicodedata.normalize("NFC", s).lower().replace("đ", "d")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", s)

corpus = pd.read_csv('data/b_corpus.csv')
questions = pd.read_csv('data/b_questions.csv')

docs = [norm(t) for t in corpus['text']]
N = len(docs)
avgdl = np.mean([len(d) for d in docs])
df = Counter()
for d in docs:
    unique_words = set(d)
    df.update(unique_words)
idf = {w: math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) for w in df}
tfs = [Counter(d) for d in docs]

k1 = 1.5
b = 0.75

results = []

for _, row in questions.iterrows():
    q = norm(row['question'])
    scores = np.zeros(N)
    for i in range(N):
        tf = tfs[i]
        score = 0.0
        for w in set(q):
            if w in tf:
                idf_w = idf.get(w, 0)
                tf_w = tf[w]
                denominator = tf_w + k1 * (1 - b + b * len(docs[i]) / avgdl)
                if denominator == 0:
                    continue
                score += idf_w * tf_w * (k1 + 1) / denominator
        scores[i] = score
    top_indices = np.argsort(-scores)[:5]
    top_ids = corpus.iloc[top_indices]['id'].values
    results.append({'id': row['id'], 'passage_ids': ' '.join(map(str, top_ids))})

submission = pd.DataFrame(results)
submission.to_csv('submission_b.csv', index=False)
