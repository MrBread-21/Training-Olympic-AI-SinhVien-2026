import pandas as pd
import numpy as np
from unicodedata import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix

def seed_all(seed):
    np.random.seed(seed)

seed_all(42)

def clean(s):
    return normalize("NFC", s).lower()

train_df = pd.read_csv('data/a_train.csv')
test_df = pd.read_csv('data/a_test.csv')

train_df['text'] = train_df['text'].apply(clean)
test_df['text'] = test_df['text'].apply(clean)

word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)

X_word = word_vectorizer.fit_transform(train_df['text'])
X_char = char_vectorizer.fit_transform(train_df['text'])
X_train = hstack([X_word, X_char])

y_train = train_df['label']

X_test_word = word_vectorizer.transform(test_df['text'])
X_test_char = char_vectorizer.transform(test_df['text'])
X_test = hstack([X_test_word, X_test_char])

clf = LogisticRegression(C=10, max_iter=3000, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = cross_val_predict(clf, X_train, y_train, cv=cv, method='predict')

print(f"Macro-F1: {f1_score(y_train, oof_pred, average='macro')}")
print("Confusion Matrix:")
print(confusion_matrix(y_train, oof_pred))

clf.fit(X_train, y_train)
test_pred = clf.predict(X_test)

submission = pd.DataFrame({'id': test_df['id'], 'label': test_pred})
submission.to_csv('submission_a.csv', index=False)
assert not submission.isna().any().any()
