
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_store(chunks):
    """
    Convert text chunks into embeddings and store them in FAISS.
    """

    if not chunks:
        return None, []

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Create a FAISS index for similarity search
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks


def search_chunks(question, index, chunks, top_k=3):
    """
    Find the most relevant chunks for a question.
    """

    if index is None or not chunks:
        return []

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    top_k = min(top_k, len(chunks))

    scores, indices = index.search(question_embedding, top_k)

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue

        results.append({
            "text": chunks[chunk_index]["text"],
            "score": float(score),
            "page": chunks[chunk_index]["page"],
            "source": chunks[chunk_index]["source"]
        })

    return results
