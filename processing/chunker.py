"""
Token-based document chunking with configurable chunk size and overlap.
Uses tiktoken for accurate token counting.
"""

import tiktoken
from config.settings import settings


class DocumentChunker:
    """Split documents into token-based chunks with overlap."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[dict]:
        """
        Split text into chunks of chunk_size tokens with chunk_overlap token overlap.

        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to each chunk (e.g., paper_id, title).

        Returns:
            List of chunk dicts with keys: 'text', 'token_count', 'chunk_index', 'metadata'
        """
        if not text.strip():
            return []

        metadata = metadata or {}

        # Tokenize the full text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            return [{
                "text": text.strip(),
                "token_count": total_tokens,
                "chunk_index": 0,
                "metadata": {**metadata, "total_chunks": 1},
            }]

        chunks = []
        start = 0
        chunk_index = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens).strip()

            if chunk_text:  # Skip empty chunks
                chunks.append({
                    "text": chunk_text,
                    "token_count": len(chunk_tokens),
                    "chunk_index": chunk_index,
                    "metadata": {**metadata},
                })
                chunk_index += 1

            # Move start forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap
            if start >= total_tokens:
                break

        # Update total_chunks in all chunk metadata
        for c in chunks:
            c["metadata"]["total_chunks"] = len(chunks)

        return chunks

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.encoding.encode(text))
