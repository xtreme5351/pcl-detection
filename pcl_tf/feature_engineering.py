import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import spacy


def build_ner_features(texts: pd.Series, model_name='en_core_web_sm') -> np.ndarray:
    """Return a (n_docs, n_entity_types) array of NER counts, normalized by token count."""
    nlp = spacy.load(model_name)
    ner_counts = []
    for doc in nlp.pipe(texts.astype(str), batch_size=64):
        ner_counts.append(Counter([ent.label_ for ent in doc.ents]))

    entity_types = sorted({t for c in ner_counts for t in c})
    mat = np.zeros((len(texts), len(entity_types)), dtype=np.float32)
    for i, c in enumerate(ner_counts):
        for j, t in enumerate(entity_types):
            mat[i, j] = c.get(t, 0)

    # normalize by token count
    token_counts = texts.str.split().apply(len).values.astype(np.float32).reshape(-1, 1)
    token_counts = np.maximum(token_counts, 1.0)
    mat = mat / token_counts
    return mat, entity_types


def build_ngram_features(texts: pd.Series, ngram_range=(1, 3), max_features=200,
                         min_df=5) -> tuple:
    """Return TF-IDF n-gram features and the fitted vectorizer."""
    vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features,
                          min_df=min_df, token_pattern=r"(?u)\b\w+\b",
                          sublinear_tf=True)
    X = vec.fit_transform(texts.astype(str)).toarray().astype(np.float32)
    return X, vec


def build_auxiliary_features(texts: pd.Series, ngram_range=(1, 3),
                             max_features=200, min_df=5):
    """Build combined NER + n-gram feature matrix. Returns (features, scaler, metadata)."""
    ner_mat, ner_types = build_ner_features(texts)
    ngram_mat, tfidf_vec = build_ngram_features(texts, ngram_range=ngram_range,
                                                 max_features=max_features, min_df=min_df)
    combined = np.concatenate([ner_mat, ngram_mat], axis=1)

    scaler = StandardScaler()
    combined = scaler.fit_transform(combined)

    metadata = {
        'ner_types': ner_types,
        'tfidf_vectorizer': tfidf_vec,
        'scaler': scaler,
        'n_ner': ner_mat.shape[1],
        'n_ngram': ngram_mat.shape[1],
        'total_dim': combined.shape[1],
    }
    return combined.astype(np.float32), metadata


def transform_auxiliary_features(texts: pd.Series, metadata: dict) -> np.ndarray:
    """Transform new texts using already-fitted NER types, TF-IDF vectorizer, and scaler."""
    nlp = spacy.load('en_core_web_sm')
    ner_counts = []
    for doc in nlp.pipe(texts.astype(str), batch_size=64):
        ner_counts.append(Counter([ent.label_ for ent in doc.ents]))

    entity_types = metadata['ner_types']
    ner_mat = np.zeros((len(texts), len(entity_types)), dtype=np.float32)
    for i, c in enumerate(ner_counts):
        for j, t in enumerate(entity_types):
            ner_mat[i, j] = c.get(t, 0)

    token_counts = texts.str.split().apply(len).values.astype(np.float32).reshape(-1, 1)
    token_counts = np.maximum(token_counts, 1.0)
    ner_mat = ner_mat / token_counts

    ngram_mat = metadata['tfidf_vectorizer'].transform(texts.astype(str)).toarray().astype(np.float32)

    combined = np.concatenate([ner_mat, ngram_mat], axis=1)
    combined = metadata['scaler'].transform(combined)
    return combined.astype(np.float32)