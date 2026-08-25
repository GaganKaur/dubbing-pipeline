"""agents/cultural_intel_agent.py — Detect cultural flags in Korean segments."""
from __future__ import annotations
import json
import logging

import vertexai
from vertexai.generative_models import GenerativeModel

from config.settings import settings
from tools.gcs_tool import save_json_artifact, load_checkpoint
from tools.vector_search_tool import VectorSearchTool
from utils.segment_models import (
    TranscribedSegment,
    CulturalFlag,
    CulturalAnalysisReport,
    FlagSeverity,
    FlagType,
)

logger = logging.getLogger(__name__)

CULTURAL_ANALYSIS_PROMPT = """
You are a {source_language} cultural linguistics expert specializing in media localization for
international audiences.

SOURCE LANGUAGE: {source_language}
TARGET AUDIENCE: international, non-{source_language}-speaking audiences (this analysis is shared
across all target languages — describe adaptation STRATEGIES, not phrasing in any
one language)

RELEVANT CULTURAL KNOWLEDGE (from knowledge base):
{rag_context}

SEGMENTS TO ANALYZE:
{segments_json}

For each segment, identify ALL cultural elements that require special handling for a
non-Korean-speaking audience.
Return EXACTLY this JSON structure:

[
  {{
    "segment_id": "seg_001",
    "flags": [
      {{
        "flag_id": "flag_seg001_001",
        "segment_id": "seg_001",
        "source_expression": "눈치",
        "flag_type": "social_norm",
        "severity": "high",
        "cultural_note": "눈치 is a uniquely Korean concept of intuitively reading social situations. No direct equivalent in most other languages.",
        "suggested_adaptation": "Convey the nuance of intuitively reading social/emotional cues — do not translate literally; use whatever natural equivalent expression exists in the target language"
      }}
    ],
    "has_high_severity": false
  }}
]

flag_type options: idiom | honorific | humor | taboo | pop_culture | historical | food | social_norm
severity options: low | medium | high

Rules:
- flag_id format: flag_{{segment_id}}_{{sequential_number}}
- Only flag elements that genuinely require cultural adaptation — do not over-flag
- has_high_severity: true if ANY flag in this segment has severity=high
- suggested_adaptation: describe the ADAPTATION STRATEGY (what nuance to preserve,
  what to avoid) — not a specific phrase in any one language, since the translation
  agent applies this strategy in whichever target language it's working in
- If no flags for a segment, return empty flags array
- Return ONLY the JSON array. No preamble, no markdown.
"""


class CulturalIntelAgent:
    """
    Analyzes Korean segments for cultural elements that need special handling.
    Uses RAG (Vertex AI Vector Search) for Korean cultural knowledge.
    Falls back gracefully if Vector Search is not configured.
    """

    def __init__(self):
        vertexai.init(project=settings.project_id, location=settings.gcp_region)
        self.model = GenerativeModel(settings.gemini_model)
        self.vector_tool = VectorSearchTool() if settings.vector_search_index_id else None

    def run(
        self,
        segments: list[TranscribedSegment],
        stem: str,
        resume: bool = False,
    ) -> dict[str, CulturalAnalysisReport]:
        """
        Analyze all segments for cultural flags.
        Returns dict mapping segment_id → CulturalAnalysisReport.
        """
        prefix = settings.source_prefix(stem)
        cached = load_checkpoint(
            settings.bucket_name, prefix, "cultural", 0, resume
        )
        if cached:
            logger.info("Loaded cultural analysis from checkpoint")
            return {
                r["segment_id"]: CulturalAnalysisReport(**r) for r in cached
            }

        # ── Batch segments to avoid context overflow ──────────────────────
        BATCH_SIZE = 10
        all_reports: dict[str, CulturalAnalysisReport] = {}

        for i in range(0, len(segments), BATCH_SIZE):
            batch = segments[i : i + BATCH_SIZE]
            logger.info(
                f"Analyzing cultural flags: segments {i+1}–{min(i+BATCH_SIZE, len(segments))}"
            )
            batch_reports = self._analyze_batch(batch)
            all_reports.update(batch_reports)

        # ── Checkpoint ────────────────────────────────────────────────────
        save_json_artifact(
            settings.bucket_name,
            f"{prefix}/cultural/cultural-iter0.json",
            [r.model_dump() for r in all_reports.values()],
        )

        high_severity_count = sum(
            1 for r in all_reports.values() if r.has_high_severity
        )
        logger.info(
            f"Cultural analysis complete. "
            f"{sum(len(r.flags) for r in all_reports.values())} flags across "
            f"{len(all_reports)} segments. "
            f"{high_severity_count} segments with high-severity flags."
        )
        return all_reports

    # ── Private ───────────────────────────────────────────────────────────

    def _analyze_batch(
        self, segments: list[TranscribedSegment]
    ) -> dict[str, CulturalAnalysisReport]:
        """Analyze one batch of segments."""
        # Build RAG context from vector search
        rag_context = self._retrieve_cultural_context(segments)

        segments_json = json.dumps(
            [
                {
                    "segment_id": s.segment_id,
                    "source_text": s.source_text,
                    "emotion_tag": s.emotion_tag,
                }
                for s in segments
            ],
            ensure_ascii=False,
            indent=2,
        )

        prompt = CULTURAL_ANALYSIS_PROMPT.format(
            source_language=settings.source_language_name,
            rag_context=rag_context,
            segments_json=segments_json,
        )

        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 8192},
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw[raw.index("\n") + 1:]  # drop opening ```[json] line
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")].rstrip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Retry each half of the batch independently to isolate the bad segment
            if len(segments) > 1:
                mid = len(segments) // 2
                logger.warning(
                    f"JSON parse failed ({exc}); retrying as two half-batches of {mid} segments"
                )
                left = self._analyze_batch(segments[:mid])
                right = self._analyze_batch(segments[mid:])
                return {**left, **right}
            logger.error(f"JSON parse failed on single segment batch; skipping. Raw:\n{raw[:500]}")
            data = []
        reports = {}
        for item in data:
            report = CulturalAnalysisReport(
                segment_id=item["segment_id"],
                flags=[CulturalFlag(**f) for f in item.get("flags", [])],
                has_high_severity=item.get("has_high_severity", False),
            )
            reports[report.segment_id] = report

        # Ensure every segment has a report (even if empty flags)
        for seg in segments:
            if seg.segment_id not in reports:
                reports[seg.segment_id] = CulturalAnalysisReport(
                    segment_id=seg.segment_id
                )

        return reports

    def _retrieve_cultural_context(self, segments: list[TranscribedSegment]) -> str:
        """Retrieve relevant cultural knowledge from Vector Search RAG."""
        if not self.vector_tool:
            return "No cultural knowledge base connected — relying on model knowledge."

        # Build a combined query from all segment texts
        combined_text = " ".join(s.source_text for s in segments)
        try:
            results = self.vector_tool.query(
                query_text=combined_text,
                top_k=settings.rag_top_k,
            )
            if not results:
                return "No relevant cultural references found in knowledge base."
            return "\n".join(
                f"- [{r['flag_type']}] {r['source_expression']}: {r['cultural_note']} → {r['target_adaptation']}"
                for r in results
            )
        except Exception as e:
            logger.warning(f"Vector Search query failed: {e} — continuing without RAG")
            return "Cultural knowledge base query failed — relying on model knowledge."
