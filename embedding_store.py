import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = []
        self.metadata = []

    def add_embeddings(self, texts, metadata_list):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.embeddings.append(embeddings)
        self.metadata.extend(metadata_list)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query, k=3):
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, k)
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

    def search_mmr(self, query, k=3, lambda_param=0.5):
        query_vector = self.model.encode([query], convert_to_numpy=True)
        if self.index is None:
            raise ValueError("Index is empty. Add embeddings first.")

        _, candidate_indices = self.index.search(query_vector, 10 * k)  # Retrieve more candidates for diversity

        selected = []
        for _ in range(k):
            best_candidate = None
            best_score = -np.inf
            for idx in candidate_indices[0]:
                if idx in selected or idx >= len(self.embeddings[0]):
                    continue
                relevance = -np.linalg.norm(query_vector - self.embeddings[0][idx])
                diversity = 0
                if selected:
                    diversity = np.min([
                        np.linalg.norm(self.embeddings[0][idx] - self.embeddings[0][sel])
                        for sel in selected
                    ])
                score = lambda_param * relevance - (1 - lambda_param) * diversity
                if score > best_score:
                    best_score = score
                    best_candidate = idx
            if best_candidate is not None:
                selected.append(best_candidate)

        results = [self.metadata[idx] for idx in selected]
        return results

    def save(self, path="vector_store"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path="vector_store"):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
