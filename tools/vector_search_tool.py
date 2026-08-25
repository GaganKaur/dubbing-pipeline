"""tools/vector_search_tool.py — Vertex AI Vector Search for cultural RAG."""
from __future__ import annotations
import json
import logging
import os
import uuid
from typing import Any

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorSearchTool:
    """
    Wraps Vertex AI Vector Search for the cultural knowledge base.

    NOTE: For the prototype, if VECTOR_SEARCH_INDEX_ID is not set,
    this tool operates in "in-memory fallback" mode — it keeps entries
    in a local dict and does simple keyword matching.
    This lets you run the full pipeline without setting up Vector Search first.
    """

    def __init__(self):
        vertexai.init(project=settings.project_id, location=settings.gcp_region)
        self._embedding_model = TextEmbeddingModel.from_pretrained(settings.embedding_model)
        self._index_id = settings.vector_search_index_id
        self._endpoint_id = settings.vector_search_endpoint_id
        self._fallback_store: list[dict] = []
        self._use_fallback = not bool(self._index_id)

        if self._use_fallback:
            logger.warning(
                "VECTOR_SEARCH_INDEX_ID not set — using in-memory fallback. "
                "Set up Vertex AI Vector Search for production."
            )

    # ── Public API ────────────────────────────────────────────────────────

    def index_entries(self, entries: list[dict]) -> int:
        """Index cultural knowledge base entries. Returns count indexed."""
        if self._use_fallback:
            self._fallback_store.extend(entries)
            return len(entries)

        # Production: embed and upsert into Vector Search
        texts = [
            f"{e['source_expression']} {e['cultural_note']} {e['target_adaptation']}"
            for e in entries
        ]
        embeddings = self._embed_batch(texts)

        # Prepare upsert datapoints
        datapoints = []
        for entry, embedding in zip(entries, embeddings):
            datapoints.append(
                {
                    "datapoint_id": str(uuid.uuid4()),
                    "feature_vector": embedding,
                    "restricts": [
                        {"namespace": "locale_pair", "allow_list": [entry.get("locale_pair", "ko-en")]},
                        {"namespace": "flag_type", "allow_list": [entry.get("flag_type", "general")]},
                    ],
                    "crowding_tag": {"crowding_attribute": entry.get("flag_type", "general")},
                    # Store full entry as numeric payload isn't native to Vector Search
                    # In production, store entry metadata in Firestore keyed by datapoint_id
                }
            )

        # Note: actual Vector Search upsert requires the aiplatform SDK
        # Full implementation depends on index type (streaming vs batch)
        # For the prototype, the fallback store is sufficient
        logger.info(f"Would upsert {len(datapoints)} datapoints to Vector Search index {self._index_id}")
        self._fallback_store.extend(entries)  # also keep in fallback for query
        return len(entries)

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-k culturally relevant entries for a query text."""
        if self._use_fallback or not self._fallback_store:
            return self._fallback_query(query_text, top_k)

        # Production path: embed query and search Vector Search endpoint
        try:
            query_embedding = self._embed_batch([query_text])[0]
            # Actual Vector Search query call here
            # Returns datapoint IDs → fetch metadata from Firestore
            # Falling back to local for prototype
            return self._fallback_query(query_text, top_k)
        except Exception as e:
            logger.warning(f"Vector Search query failed: {e}")
            return self._fallback_query(query_text, top_k)

    # ── Private ───────────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using gemini-embedding-001.

        gemini-embedding-001 accepts one input per request (no batch
        endpoint yet), unlike the older gecko-based embedding models —
        so this loops instead of chunking into multi-instance calls.
        """
        all_embeddings = []
        for text in texts:
            text_input = TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT")
            result = self._embedding_model.get_embeddings([text_input])
            all_embeddings.append(result[0].values)
        return all_embeddings

    def _fallback_query(self, query_text: str, top_k: int) -> list[dict]:
        """
        Simple keyword-based fallback when Vector Search is not available.
        Scores entries by how many query words appear in source_expression or cultural_note.
        """
        if not self._fallback_store:
            return []

        query_words = set(query_text.lower().split())
        scored = []
        for entry in self._fallback_store:
            searchable = (
                (entry.get("source_expression") or "") + " " +
                (entry.get("cultural_note") or "") + " " +
                (entry.get("target_adaptation") or "")
            ).lower()
            score = sum(1 for w in query_words if w in searchable)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return [entry for _, entry in scored[:top_k]]
