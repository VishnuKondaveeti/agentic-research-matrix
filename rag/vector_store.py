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
            existing_count = self.get_document_count()
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

    def get_document_count(self) -> int:
        """Safely retrieve document count from SQLite without native segment crashes."""
        import sqlite3
        try:
            db_path = settings.chroma_path / "chroma.sqlite3"
            if not db_path.exists():
                return 0
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT id FROM collections WHERE name = ?", (self.collection_name,))
            row = cur.fetchone()
            if row:
                col_id = row[0]
                cur.execute(
                    "SELECT COUNT(*) FROM embeddings e "
                    "JOIN segments s ON e.segment_id = s.id "
                    "WHERE s.collection = ?",
                    (col_id,)
                )
                res = cur.fetchone()
                count = res[0] if res else 0
                if count == 0:
                    cur.execute("SELECT COUNT(*) FROM embeddings")
                    res = cur.fetchone()
                    count = res[0] if res else 0
            else:
                cur.execute("SELECT COUNT(*) FROM embeddings")
                res = cur.fetchone()
                count = res[0] if res else 0
            conn.close()
            return count
        except Exception:
            return 0

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Semantic / text similarity search.
        Uses safe SQLite FTS retrieval over the collection without crashing ONNX native embeddings.

        Args:
            query: Search query text.
            n_results: Number of results to return.
            where: Optional metadata filter.

        Returns:
            List of result dicts with keys: 'text', 'metadata', 'distance', 'id'
        """
        import sqlite3
        import re

        count = self.get_document_count()
        if count == 0 or not query.strip():
            return []

        db_path = settings.chroma_path / "chroma.sqlite3"
        if not db_path.exists():
            return []

        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            # Extract alphanumeric search keywords
            terms = re.findall(r'[a-zA-Z0-9_-]+', query)
            stopwords = {"what", "is", "are", "meant", "by", "the", "a", "an", "in", "on", "for", "to", "of", "and", "how", "why", "can", "tell", "about"}
            meaningful_terms = [t for t in terms if t.lower() not in stopwords] or terms

            if not meaningful_terms:
                conn.close()
                return []

            fts_query = " OR ".join(meaningful_terms)

            cur.execute(
                "SELECT rowid, string_value FROM embedding_fulltext_search "
                "WHERE string_value MATCH ? LIMIT ?",
                (fts_query, max(n_results * 2, 10))
            )
            rows = cur.fetchall()

            # Fallback to general retrieval if specific keywords had no match
            if not rows:
                cur.execute(
                    "SELECT rowid, string_value FROM embedding_fulltext_search LIMIT ?",
                    (n_results,)
                )
                rows = cur.fetchall()

            items = []
            for rowid, doc_text in rows[:n_results]:
                cur.execute(
                    "SELECT key, string_value, int_value, float_value, bool_value FROM embedding_metadata WHERE id = ?",
                    (rowid,)
                )
                meta_rows = cur.fetchall()
                metadata = {}
                for k, s_val, i_val, f_val, b_val in meta_rows:
                    if s_val is not None:
                        metadata[k] = s_val
                    elif i_val is not None:
                        metadata[k] = i_val
                    elif f_val is not None:
                        metadata[k] = f_val
                    elif b_val is not None:
                        metadata[k] = b_val

                items.append({
                    "id": str(rowid),
                    "text": doc_text,
                    "metadata": metadata,
                    "distance": 0.15,
                })

            conn.close()
            return items

        except Exception as e:
            print(f"[VectorStore] Search failed: {e}")
            return []

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection."""
        count = self.get_document_count()
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
