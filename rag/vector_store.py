"""
ChromaDB vector store wrapper.
Handles collection management, document storage, and similarity search.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from config.settings import settings


class VectorStore:
    """ChromaDB-backed vector store for research paper embeddings."""

    def __init__(self, collection_name: str = "research_papers"):
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """
        Add documents to the vector store.

        Args:
            texts: List of text chunks.
            metadatas: Optional metadata for each chunk.
            ids: Optional unique IDs for each chunk.

        Returns:
            Number of documents added.
        """
        if not texts:
            return 0

        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

        # Clean metadata - ChromaDB only supports str, int, float, bool
        if metadatas:
            metadatas = [self._clean_metadata(m) for m in metadatas]

        # Batch upsert (ChromaDB handles embedding internally with default model)
        batch_size = 100
        total_added = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size] if metadatas else None

            self.collection.upsert(
                documents=batch_texts,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            total_added += len(batch_texts)

        return total_added

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Semantic similarity search.

        Args:
            query: Search query text.
            n_results: Number of results to return.
            where: Optional metadata filter.

        Returns:
            List of result dicts with keys: 'text', 'metadata', 'distance', 'id'
        """
        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, self.collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            print(f"[VectorStore] Search failed: {e}")
            return []

        items = []
        if results and results.get("documents") is not None and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                items.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if (results.get("metadatas") is not None and len(results["metadatas"]) > 0) else {},
                    "distance": results["distances"][0][i] if (results.get("distances") is not None and len(results["distances"]) > 0) else 0,
                    "id": results["ids"][0][i] if (results.get("ids") is not None and len(results["ids"]) > 0) else "",
                })

        return items

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(settings.chroma_path),
        }

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_embeddings_3d(self, limit: int = 100) -> dict:
        """Get 3D dimensionality reduction for visualization."""
        try:
            results = self.collection.get(
                include=["embeddings", "metadatas", "documents"],
                limit=limit,
            )
            
            if not results or results.get("embeddings") is None or len(results["embeddings"]) == 0:
                return {"status": "error", "message": "No embeddings found."}
            
            embeddings = np.array(results["embeddings"])
            
            # Use PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            points_3d = pca.fit_transform(embeddings)
            
            points = []
            for i in range(len(points_3d)):
                meta = results["metadatas"][i] if results.get("metadatas") is not None else {}
                points.append({
                    "x": float(points_3d[i][0]),
                    "y": float(points_3d[i][1]),
                    "z": float(points_3d[i][2]),
                    "id": results["ids"][i],
                    "title": meta.get("title", "Untitled Fragment"),
                    "text": results["documents"][i][:200] + "..." if results.get("documents") is not None else ""
                })
            
            return {
                "status": "success",
                "points": points
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _clean_metadata(self, metadata: dict) -> dict:
        """Clean metadata to only contain ChromaDB-compatible types."""
        cleaned = {}
        # Ensure np is imported
        import numpy as np
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            elif isinstance(v, list):
                cleaned[k] = ", ".join(str(item) for item in v)
            elif v is None:
                cleaned[k] = ""
            else:
                cleaned[k] = str(v)
        return cleaned
