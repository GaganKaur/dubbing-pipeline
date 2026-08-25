"""agents/translation_agent.py — Culturally-aware Korean → target-language translation."""
from __future__ import annotations
import json
import logging
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel

from config.settings import settings
from tools.gcs_tool import save_json_artifact
from utils.segment_models import (
    TranscribedSegment,
    CulturalAnalysisReport,
    TranslatedSegment,
    ConsensusQAReport,
)

logger = logging.getLogger(__name__)

_STYLE_GUIDES_DIR = Path(__file__).parent.parent / "prompts/style_guides"

TRANSLATION_PROMPT = """
You are a professional {source_language}-to-{target_language} subtitle translator specializing in media localization.

STYLE GUIDE:
{style_guide}

CULTURAL FLAGS (MUST respect these adaptations):
{cultural_flags}

{critique_section}

SEGMENTS TO TRANSLATE:
{segments_json}

Translate each Korean segment to {target_language} for subtitles.
Return EXACTLY this JSON array:

[
  {{
    "segment_id": "seg_001",
    "translated_text": "Hello, my name is Kim Minjun.",
    "cultural_adaptations_applied": ["Kept Korean name order", "Preserved formal register"]
  }}
]

Rules:
- translated_text: the final {target_language} subtitle text (ready for display)
- Max 42 characters per line, max 2 lines — use \\n for line breaks
- cultural_adaptations_applied: list the specific adaptations you made (for QA audit trail)
- If a cultural flag suggests an adaptation, APPLY IT — do not translate literally
- Preserve the speaker's register (formal/casual) as noted in the style guide
- Return ONLY the JSON array. No preamble, no markdown.
"""

CRITIQUE_SECTION_TEMPLATE = """
QUALITY CONTROL CRITIQUE FROM PREVIOUS ITERATION (iteration {iteration}):
The following issues were found. Your translation MUST address all of them:

{issues}

Focus especially on SEVERE and MODERATE issues — these are blocking.
"""


class TranslationAgent:
    """
    Translates Korean segments to the configured target language with cultural awareness.
    Accepts QA critique objects to improve on previous iterations.
    """

    def __init__(self):
        vertexai.init(project=settings.project_id, location=settings.gcp_region)
        self.model = GenerativeModel(settings.gemini_model)
        guide_path = _STYLE_GUIDES_DIR / f"{settings.source_lang}_source.txt"
        if not guide_path.exists():
            guide_path = _STYLE_GUIDES_DIR / "korean_source.txt"
            logger.warning(f"No style guide for source lang '{settings.source_lang}', falling back to Korean")
        self._style_guide = guide_path.read_text(encoding="utf-8").format(
            target_language=settings.target_language_name
        )

    def run(
        self,
        segments: list[TranscribedSegment],
        cultural_reports: dict[str, CulturalAnalysisReport],
        stem: str,
        iteration: int = 0,
        qa_critiques: dict[str, ConsensusQAReport] | None = None,
    ) -> list[TranslatedSegment]:
        """
        Translate segments. On iteration > 0, qa_critiques contains
        the per-segment QA issues to address.

        Returns list of TranslatedSegment.
        """
        logger.info(
            f"Translation iteration {iteration}: "
            f"{len(segments)} segments"
            + (f" with {sum(len(r.issues) for r in qa_critiques.values())} QA issues to fix"
               if qa_critiques else "")
        )

        # Only re-translate segments that failed QA (or all on first pass).
        # qa_critiques (built by the orchestrator) contains only failed
        # segments, so membership alone identifies what needs re-translation.
        if qa_critiques and iteration > 0:
            segments_to_translate = [s for s in segments if s.segment_id in qa_critiques]
            logger.info(
                f"Re-translating {len(segments_to_translate)} failed segments "
                f"(skipping {len(segments) - len(segments_to_translate)} passed)"
            )
        else:
            segments_to_translate = segments

        BATCH_SIZE = 10
        all_translations: list[TranslatedSegment] = []

        for i in range(0, len(segments_to_translate), BATCH_SIZE):
            batch = segments_to_translate[i : i + BATCH_SIZE]
            batch_critiques = (
                {s.segment_id: qa_critiques[s.segment_id]
                 for s in batch if s.segment_id in (qa_critiques or {})}
                if qa_critiques else {}
            )
            translations = self._translate_batch(
                batch, cultural_reports, iteration, batch_critiques
            )
            all_translations.extend(translations)

        # Save checkpoint
        prefix = settings.output_prefix(stem)
        save_json_artifact(
            settings.bucket_name,
            f"{prefix}/trans/trans-iter{iteration}.json",
            [t.model_dump() for t in all_translations],
        )

        return all_translations

    # ── Private ───────────────────────────────────────────────────────────

    def _translate_batch(
        self,
        segments: list[TranscribedSegment],
        cultural_reports: dict[str, CulturalAnalysisReport],
        iteration: int,
        qa_critiques: dict[str, ConsensusQAReport],
    ) -> list[TranslatedSegment]:

        # Build cultural flags context for this batch
        cultural_flags_text = self._format_cultural_flags(segments, cultural_reports)

        # Build critique section if we have QA feedback
        critique_section = ""
        if qa_critiques:
            issues_text = []
            for seg_id, report in qa_critiques.items():
                for issue in report.issues:
                    issues_text.append(
                        f"  [{issue.severity.upper()}] Segment {seg_id} — "
                        f"{issue.issue_type}: {issue.description}\n"
                        f"  FIX: {issue.fix_hint}"
                    )
            if issues_text:
                critique_section = CRITIQUE_SECTION_TEMPLATE.format(
                    iteration=iteration - 1,
                    issues="\n\n".join(issues_text),
                )

        segments_json = json.dumps(
            [
                {
                    "segment_id": s.segment_id,
                    "source_text": s.source_text,
                    "emotion_tag": s.emotion_tag,
                    "speaker_id": s.speaker_id,
                }
                for s in segments
            ],
            ensure_ascii=False,
            indent=2,
        )

        prompt = TRANSLATION_PROMPT.format(
            source_language=settings.source_language_name,
            target_language=settings.target_language_name,
            style_guide=self._style_guide,
            cultural_flags=cultural_flags_text,
            critique_section=critique_section,
            segments_json=segments_json,
        )

        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 8192},
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw[raw.index("\n") + 1:]
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")].rstrip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            if len(segments) > 1:
                mid = len(segments) // 2
                logger.warning(
                    f"JSON parse failed ({exc}); retrying as two half-batches of {mid} segments"
                )
                left = self._translate_batch(segments[:mid], cultural_reports, iteration, qa_critiques)
                right = self._translate_batch(segments[mid:], cultural_reports, iteration, qa_critiques)
                return left + right
            logger.error(f"JSON parse failed on single-segment batch; skipping. Raw:\n{raw[:500]}")
            data = []

        # Build lookup from source segments for timestamps
        seg_lookup = {s.segment_id: s for s in segments}

        results = []
        for item in data:
            seg = seg_lookup[item["segment_id"]]
            results.append(
                TranslatedSegment(
                    segment_id=seg.segment_id,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    speaker_id=seg.speaker_id,
                    source_text=seg.source_text,
                    translated_text=item["translated_text"],
                    cultural_adaptations_applied=item.get(
                        "cultural_adaptations_applied", []
                    ),
                    iteration=iteration,
                )
            )
        return results

    def _format_cultural_flags(
        self,
        segments: list[TranscribedSegment],
        cultural_reports: dict[str, CulturalAnalysisReport],
    ) -> str:
        """Format cultural flags as readable context for the translation prompt."""
        lines = []
        for seg in segments:
            report = cultural_reports.get(seg.segment_id)
            if report and report.flags:
                for flag in report.flags:
                    lines.append(
                        f"  Segment {seg.segment_id} | [{flag.flag_type} / {flag.severity}] "
                        f'"{flag.source_expression}": {flag.cultural_note} '
                        f"→ ADAPT AS: {flag.suggested_adaptation}"
                    )
        return "\n".join(lines) if lines else "No special cultural flags for these segments."
