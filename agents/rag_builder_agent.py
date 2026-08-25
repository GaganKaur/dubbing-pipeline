"""agents/rag_builder_agent.py — Cultural RAG builder using Gemini structured output."""
from __future__ import annotations
import json
import logging

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from config.settings import settings
from tools.gcs_tool import save_json_artifact
from tools.vector_search_tool import VectorSearchTool

logger = logging.getLogger(__name__)

CATEGORIES = {
    "idiom": "Korean idioms with English equivalents used in dramas and films",
    "honorific": "Korean honorific titles like 오빠 언니 선생님 사장님 and subtitle handling",
    "social_norm": "Korean social concepts like 눈치 정 한 체면 with no English equivalent",
    "humor": "Korean wordplay and comedy patterns in media and how to adapt them",
    "food": "Korean dishes and drinks in media needing context for English viewers",
    "pop_culture": "Korean entertainment slang and idol culture terms",
    "historical": "Korean historical and political references needing English context",
    "taboo": "Culturally sensitive phrases between Korean and English speaking audiences",
}

# Structured output schema — Gemini guarantees valid JSON matching this
RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "source_expression": {"type": "string"},
            "cultural_note": {"type": "string"},
            "target_adaptation": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["source_expression", "cultural_note", "target_adaptation", "severity"],
    },
}

ENTRY_PROMPT = """You are a Korean-English media localization expert.

List 5 important {category_name} examples for: {category_description}

For each entry:
- source_expression: the Korean word or short phrase
- cultural_note: one sentence explaining the challenge for English audiences
- target_adaptation: one sentence on how to handle it in English subtitles
- severity: low, medium, or high"""


class RAGBuilderAgent:
    def __init__(self):
        vertexai.init(project=settings.project_id, location=settings.gcp_region)
        self.model = GenerativeModel(settings.gemini_model)
        self.vector_tool = VectorSearchTool()

    def run(self, num_passes: int = 2) -> int:
        logger.info(
            f"Building cultural RAG for ko→en "
            f"({len(CATEGORIES)} categories × {num_passes} passes)..."
        )
        all_entries = []

        for pass_num in range(1, num_passes + 1):
            for category_name, category_desc in CATEGORIES.items():
                logger.info(f"Pass {pass_num}/{num_passes} — {category_name}...")
                entries = self._generate_category(category_name, category_desc)
                # Stamp category and locale onto each entry
                for e in entries:
                    e["category"] = category_name
                    e["flag_type"] = category_name
                    e["locale_pair"] = "ko-en"
                all_entries.extend(entries)
                logger.info(f"  → {len(entries)} entries")

        # Deduplicate by source_expression
        seen: set[str] = set()
        deduped = []
        for entry in all_entries:
            key = entry.get("source_expression", "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(entry)

        logger.info(f"Total after dedup: {len(deduped)} entries")

        save_json_artifact(settings.bucket_name, "rag/ko_en_cultural_kb.json", deduped)
        logger.info(f"Saved to gs://{settings.bucket_name}/rag/ko_en_cultural_kb.json")

        indexed_count = self.vector_tool.index_entries(deduped)
        logger.info(f"RAG build complete. {indexed_count} entries ready.")
        return indexed_count

    def _generate_category(self, category_name: str, category_desc: str) -> list[dict]:
        prompt = ENTRY_PROMPT.format(
            category_name=category_name,
            category_description=category_desc,
        )
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.6,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
            entries = json.loads(response.text)
            if not isinstance(entries, list):
                raise ValueError(f"Expected list, got {type(entries)}")
            return entries
        except Exception as e:
            logger.warning(f"  Failed for {category_name}: {e}")
            return []
