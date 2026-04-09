"""Local vector store for scene embeddings using numpy."""

import json
from pathlib import Path

import numpy as np


class EmbeddingStore:
    """Save, load, search, and cluster scene embeddings."""

    def __init__(self, vectors: np.ndarray | None = None, metadata: list[dict] | None = None):
        self.vectors = vectors if vectors is not None else np.empty((0, 0))
        self.metadata = metadata or []

    @classmethod
    def from_embeddings(cls, embeddings: list[dict]) -> "EmbeddingStore":
        """Create a store from the output of embed_scenes()."""
        if not embeddings:
            return cls()

        vectors = np.array([e["vector"] for e in embeddings], dtype=np.float32)
        metadata = [
            {
                "scene_index": e["scene_index"],
                "dimensions": e["dimensions"],
                "modalities": e["modalities"],
            }
            for e in embeddings
        ]
        return cls(vectors=vectors, metadata=metadata)

    def save(self, path: Path) -> None:
        """Save embeddings to .npz + .json metadata sidecar."""
        path = Path(path)
        np.savez_compressed(path, vectors=self.vectors)

        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def load(cls, path: Path) -> "EmbeddingStore":
        """Load embeddings from .npz + .json."""
        path = Path(path)
        data = np.load(path)
        vectors = data["vectors"]

        meta_path = path.with_suffix(".json")
        metadata = json.loads(meta_path.read_text())

        return cls(vectors=vectors, metadata=metadata)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """Find the top-k most similar scenes by cosine similarity.

        Returns:
            List of dicts with 'scene_index', 'score', and 'modalities'.
        """
        if len(self.vectors) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
        similarities = _cosine_similarity(query, self.vectors)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "scene_index": self.metadata[idx]["scene_index"],
                "score": float(similarities[idx]),
                "modalities": self.metadata[idx]["modalities"],
            })
        return results

    def find_duplicates(self, threshold: float = 0.95) -> list[tuple[int, int, float]]:
        """Find scene pairs with cosine similarity above threshold.

        Returns:
            List of (scene_index_a, scene_index_b, similarity) tuples.
        """
        if len(self.vectors) < 2:
            return []

        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = self.vectors / norms
        sim_matrix = normalized @ normalized.T

        duplicates = []
        n = len(self.vectors)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    duplicates.append((
                        self.metadata[i]["scene_index"],
                        self.metadata[j]["scene_index"],
                        float(sim_matrix[i, j]),
                    ))
        return duplicates

    def cluster(self, n_clusters: int) -> dict[int, list[int]]:
        """Cluster scenes using K-means.

        Returns:
            Dict mapping cluster_id to list of scene indices.
        """
        if len(self.vectors) < n_clusters:
            return {0: [m["scene_index"] for m in self.metadata]}

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.vectors)

        clusters: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            cluster_id = int(label)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(self.metadata[i]["scene_index"])
        return clusters


def _cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(len(vectors))

    vec_norms = np.linalg.norm(vectors, axis=1)
    vec_norms = np.maximum(vec_norms, 1e-10)

    return (vectors @ query) / (vec_norms * query_norm)
