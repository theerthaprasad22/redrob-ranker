"""
embeddings.py — pluggable semantic encoder.

Two backends, same interface:

  * "lsa"  (default): TfidfVectorizer + TruncatedSVD (latent semantic analysis).
            100% local, no model download, no network, milliseconds per 1K docs
            on CPU. Captures latent semantics so paraphrased / plain-language
            profiles still match the role meaning. This is the backend used in
            the offline-safe default pipeline and in CI.

  * "st"   (optional upgrade): sentence-transformers (e.g. BAAI/bge-small-en-v1.5
            or all-MiniLM-L6-v2). Higher-quality embeddings. The model must be
            cached locally beforehand (see precompute.py / README) because the
            ranking step runs with NO network. To stay within the 5-min CPU
            budget the pipeline only ST-encodes the retrieval shortlist, not the
            full 100K pool.

Both return L2-normalised float32 vectors, so cosine similarity == dot product.
"""

from __future__ import annotations

import numpy as np


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class LSAEncoder:
    """TF-IDF + Truncated SVD. Fit on the corpus, then transform any text."""

    def __init__(self, n_components: int = 256, max_features: int = 60000):
        self.n_components = n_components
        self.max_features = max_features
        self._vec = None
        self._svd = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> "LSAEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        # min_df=2 helps on large corpora but empties the vocab on tiny ones,
        # so adapt it (and fall back to 1 if pruning leaves nothing).
        min_df = 2 if len(corpus) >= 20 else 1
        try:
            self._vec = TfidfVectorizer(
                max_features=self.max_features, ngram_range=(1, 2),
                sublinear_tf=True, min_df=min_df, stop_words="english",
            )
            X = self._vec.fit_transform(corpus)
        except ValueError:
            self._vec = TfidfVectorizer(
                max_features=self.max_features, ngram_range=(1, 2),
                sublinear_tf=True, min_df=1,
            )
            X = self._vec.fit_transform(corpus)
        # SVD components must be < min(n_samples, n_features).
        n_comp = int(min(self.n_components, max(2, min(X.shape) - 1)))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(X)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LSAEncoder.encode called before fit()")
        X = self._vec.transform(texts)
        Z = self._svd.transform(X)
        return _normalize(Z)


class STEncoder:
    """sentence-transformers wrapper (optional). Loads from local cache only."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 max_chars: int = 1200, batch_size: int = 64):
        self.model_name = model_name
        self.max_chars = max_chars
        self.batch_size = batch_size
        self._model = None

    def fit(self, corpus: list[str]) -> "STEncoder":
        # No corpus fitting needed; lazily load the model here.
        from sentence_transformers import SentenceTransformer  # noqa: F401
        import os
        # Force offline so a stray network call can't happen at ranking time.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device="cpu")
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self.fit([])
        texts = [t[: self.max_chars] for t in texts]
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)


def make_encoder(kind: str = "lsa", **kwargs):
    kind = (kind or "lsa").lower()
    if kind in ("st", "sbert", "sentence-transformers"):
        return STEncoder(**{k: v for k, v in kwargs.items()
                            if k in ("model_name", "max_chars", "batch_size")})
    return LSAEncoder(**{k: v for k, v in kwargs.items()
                        if k in ("n_components", "max_features")})


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of `a` against a single vector `b`.

    Both are expected L2-normalised, so this is just a dot product. Returns
    values mapped from [-1, 1] to [0, 1].
    """
    sims = a @ b.reshape(-1)
    return (sims + 1.0) / 2.0
