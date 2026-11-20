import math, random, re, unicodedata
from collections import Counter
import numpy as np, pandas as pd

def seed_all(s=42):
    random.seed(s); np.random.seed(s)
seed_all(42)

def tok(s):
    """NFC, chữ thường, bỏ dấu (câu hỏi có thể gõ không dấu), tách từ."""
    s = unicodedata.normalize("NFC", str(s)).lower().replace("đ", "d")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return re.findall(r"\w+", s)

corpus, q = pd.read_csv("data/b_corpus.csv"), pd.read_csv("data/b_questions.csv")
docs = [tok(t) for t in corpus.text]
N, avgdl = len(docs), np.mean([len(d) for d in docs])
df = Counter(w for d in docs for w in set(d))
idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
tfs = [Counter(d) for d in docs]

def bm25(query, k1=1.5, b=0.75):
    s = np.zeros(N)
    for w in set(tok(query)):
        if w not in idf: continue
        for i, tf in enumerate(tfs):
            f = tf.get(w, 0)
            if f: s[i] += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(docs[i]) / avgdl))
    return s

top = [np.argsort(-bm25(t))[:5] for t in q.question]
pd.DataFrame({"id": q.id, "passage_ids": [" ".join(corpus.id[j] for j in r) for r in top]}).to_csv("submission_b.csv", index=False)
print("đã ghi submission_b.csv", len(q), "câu hỏi")
