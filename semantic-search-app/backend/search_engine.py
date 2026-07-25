"""
search_engine.py
-----------------
Core semantic similarity search engine, ported directly from the original
Colab notebook logic (Word2Vec + cosine similarity), wrapped into a
reusable class the FastAPI backend can call.

Pipeline:
  1. Load a CSV with Title / Content / Keywords columns
  2. Clean + combine text  ->  combined_text
  3. Tokenize
  4. Train a Word2Vec (skip-gram) model on the corpus
  5. Build a document embedding for every row (mean of word vectors)
  6. At query time: embed the query the same way, rank docs by cosine similarity
"""

from __future__ import annotations

import re
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = ["title", "content", "keywords"]

# Where trained artifacts get cached so the server doesn't need to
# retrain every time it restarts.
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "word2vec.model"
EMBEDDINGS_PATH = ARTIFACT_DIR / "doc_embeddings.npy"
DATAFRAME_PATH = ARTIFACT_DIR / "dataframe.pkl"


def clean_text(text) -> str:
    """Lowercase, strip URLs/HTML/punctuation, collapse whitespace."""
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # URLs
    text = re.sub(r"<.*?>", "", text)  # HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # punctuation/special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text.split()


class SemanticSearchEngine:
    """Holds a trained Word2Vec model + document embeddings for one dataset."""

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.model: Optional[Word2Vec] = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self.vector_size = 100

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit_from_dataframe(self, df: pd.DataFrame) -> dict:
        """Run the full pipeline on a raw dataframe and train the model."""
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s): {', '.join(missing)}. "
                f"Your CSV needs: Title, Content, Keywords."
            )

        for col in REQUIRED_COLUMNS:
            df[col] = df[col].fillna("")

        df["clean_title"] = df["title"].apply(clean_text)
        df["clean_content"] = df["content"].apply(clean_text)
        df["clean_keywords"] = df["keywords"].apply(clean_text)

        df["combined_text"] = (
            df["clean_title"] + ". " + df["clean_keywords"] + ". " + df["clean_content"]
        )
        df["tokens"] = df["combined_text"].apply(tokenize)

        # Drop rows that produced no tokens at all (empty row safety net)
        df = df[df["tokens"].map(len) > 0].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No usable text found after cleaning — check your CSV content.")

        model = Word2Vec(
            sentences=df["tokens"],
            vector_size=self.vector_size,
            window=5,
            min_count=1,  # min_count=1 so small demo datasets still work
            workers=4,
            sg=1,  # skip-gram, better for semantics
            epochs=20,
        )

        doc_embeddings = np.vstack(
            df["tokens"].apply(lambda toks: self._document_vector(toks, model)).values
        )

        self.df = df
        self.model = model
        self.doc_embeddings = doc_embeddings
        self._persist()

        return {
            "num_documents": len(df),
            "vocab_size": len(model.wv.index_to_key),
            "vector_size": self.vector_size,
        }

    def _document_vector(self, tokens: list[str], model: Word2Vec) -> np.ndarray:
        vectors = [model.wv[w] for w in tokens if w in model.wv]
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    def _query_vector(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
        if not vectors:
            return np.zeros(self.vector_size).reshape(1, -1)
        return np.mean(vectors, axis=0).reshape(1, -1)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 5) -> dict:
        if self.model is None or self.doc_embeddings is None or self.df is None:
            raise RuntimeError("No model loaded yet. Upload a dataset first.")

        start = time.perf_counter()
        q_vec = self._query_vector(query)
        similarities = cosine_similarity(q_vec, self.doc_embeddings)[0]

        top_k = max(1, min(top_k, len(self.df)))
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append(
                {
                    "title": row.get("title", ""),
                    "content": row.get("content", ""),
                    "keywords": row.get("keywords", ""),
                    "score": float(similarities[idx]),
                }
            )

        took_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"query": query, "results": results, "took_ms": took_ms}

    # ------------------------------------------------------------------ #
    # Persistence (so a server restart doesn't force a retrain)
    # ------------------------------------------------------------------ #
    def _persist(self) -> None:
        self.model.save(str(MODEL_PATH))
        np.save(EMBEDDINGS_PATH, self.doc_embeddings)
        with open(DATAFRAME_PATH, "wb") as f:
            pickle.dump(self.df, f)

    def load_from_disk(self) -> bool:
        if not (MODEL_PATH.exists() and EMBEDDINGS_PATH.exists() and DATAFRAME_PATH.exists()):
            return False
        self.model = Word2Vec.load(str(MODEL_PATH))
        self.doc_embeddings = np.load(EMBEDDINGS_PATH)
        with open(DATAFRAME_PATH, "rb") as f:
            self.df = pickle.load(f)
        return True

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.doc_embeddings is not None

    @property
    def num_documents(self) -> int:
        return 0 if self.df is None else len(self.df)
