from __future__ import annotations

import numpy as np
from gensim.models import Word2Vec


def train_word2vec(
    texts: list[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 1,
    sg: int = 1,
    epochs: int = 20,
) -> Word2Vec:
    # The cleaning pipeline returns strings, so we tokenize here with a simple whitespace split.
    tokenized_texts = [text.split() for text in texts]
    return Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )


def vectorize_texts(texts: list[str], model: Word2Vec) -> np.ndarray:
    vector_size = model.wv.vector_size
    vectors = []

    for text in texts:
        tokens = [token for token in text.split() if token in model.wv]
        if not tokens:
            # Fall back to zeros when a text has no in-vocabulary tokens after preprocessing.
            vectors.append(np.zeros(vector_size, dtype=np.float32))
            continue

        # Represent each text by the mean of its token embeddings.
        token_vectors = np.array([model.wv[token] for token in tokens], dtype=np.float32)
        vectors.append(token_vectors.mean(axis=0))

    return np.vstack(vectors)
