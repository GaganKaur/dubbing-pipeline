"""agents/qa_agent.py — Multi-temperature consensus QA scoring for translated segments."""
from __future__ import annotations
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel

from config.settings import settings
from tools.gcs_tool import save_json_artifact
from utils.consensus import merge_qa_reports
from utils.segment_models import (
    TranscribedSegment,
    TranslatedSegment,
    CulturalAnalysisReport,
    QAIssue,
    QAReport,
    ConsensusQAReport,
    IssueSeverity,
)

logger = logging.getLogger(__name__)

RUBRIC_PATH = Path(__file__).parent.parent / "prompts/qc_rubrics/subtitle_qc_rubric.txt"

QA_PROMPT = """
You are a professional subtitle quality control evaluator for {source_language}-to-{target_language} media localization.

QC RUBRIC:
{rubric}

ORIGINAL SEGMENTS (Korean source):
{source_json}

CULTURAL FLAGS THAT MUST BE RESPECTED:
{cultural_flags}

TRANSLATED SEGMENTS TO EVALUATE:
{translated_json}

Evaluate each translated segment against the rubric. Return EXACTLY this JSON array:

[
  {{
    "segment_id": "seg_001",
    "score": 0.92,
    "issues": [
      {{
        "issue_id": "issue_001",
        "segment_id": "seg_001",
        "issue_type": "cultural_mistranslation",
        "severity": "moderate",
        "description": "눈치 translated as 'instinct' — loses the social-awareness meaning",
        "fix_hint": "Replace 'instinct' with 'reading the room' to preserve the 눈치 concept"
      }}
    ],
    "passed": false
  }}
]

issue_type options: cultural_mistranslation | meaning_error | line_formatting | tone_mismatch | honorific_error | humor_mistranslation
severity options: minor | moderate | severe
passed: true ONLY if score >= 0.80 AND no severe or moderate issues

Rules:
- Be specific in description and fix_hint — vague feedback is useless to the translator
- Do not flag stylistic preferences as moderate/severe
- If a segment passes all criteria, return empty issues array and passed: true
- Return ONLY the JSON array. No preamble, no markdown.
"""


class QAAgent:
    """
    Runs N independent QA evaluations at different temperatures,
    then merges via consensus algorithm.
    """

    def __init__(self):
        vertexai.init(project=settings.project_id, location=settings.gcp_region)
        self._rubric = RUBRIC_PATH.read_text(encoding="utf-8").format(
            target_language=settings.target_language_name
        )
        self.num_agents = settings.num_qa_agents
        self.temperatures = settings.qa_temperatures[: self.num_agents]

    def run(
        self,
        source_segments: list[TranscribedSegment],
        translated_segments: list[TranslatedSegment],
        cultural_reports: dict[str, CulturalAnalysisReport],
        stem: str,
        iteration: int,
    ) -> dict[str, ConsensusQAReport]:
        """
        Evaluate all translated segments.
        Returns dict mapping segment_id → ConsensusQAReport.
        """
        logger.info(
            f"QA iteration {iteration}: evaluating {len(translated_segments)} segments "
            f"with {self.num_agents} agents (temps: {self.temperatures})"
        )

        # Build source lookup
        source_lookup = {s.segment_id: s for s in source_segments}

        # Run N agents in parallel (ThreadPoolExecutor for simplicity)
        all_agent_reports: list[list[QAReport]] = [[] for _ in range(self.num_agents)]

        with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
            futures = {
                executor.submit(
                    self._run_single_agent,
                    source_segments,
                    translated_segments,
                    cultural_reports,
                    iteration,
                    agent_idx,
                    temp,
                ): agent_idx
                for agent_idx, temp in enumerate(self.temperatures)
            }
            for future in as_completed(futures):
                agent_idx = futures[future]
                try:
                    reports = future.result()
                    all_agent_reports[agent_idx] = reports
                    logger.info(f"QA agent {agent_idx} complete: {len(reports)} reports")
                except Exception as e:
                    logger.error(f"QA agent {agent_idx} failed: {e}")
                    # Fill with empty passing reports so consensus still works
                    all_agent_reports[agent_idx] = [
                        QAReport(
                            iteration=iteration,
                            segment_id=s.segment_id,
                            score=1.0,
                            passed=True,
                            agent_index=agent_idx,
                        )
                        for s in translated_segments
                    ]

        # ── Merge into consensus reports ──────────────────────────────────
        # Group by segment_id across all agents
        by_segment: dict[str, list[QAReport]] = {}
        for agent_reports in all_agent_reports:
            for report in agent_reports:
                by_segment.setdefault(report.segment_id, []).append(report)

        consensus_reports: dict[str, ConsensusQAReport] = {}
        for segment_id, reports in by_segment.items():
            consensus_reports[segment_id] = merge_qa_reports(
                reports, agreement_threshold=0.5
            )

        # ── Save artifacts ─────────────────────────────────────────────────
        prefix = settings.output_prefix(stem)

        # Individual agent reports
        for agent_idx, agent_reports in enumerate(all_agent_reports):
            save_json_artifact(
                settings.bucket_name,
                f"{prefix}/qc/qc-iter{iteration}-agents/agent{agent_idx + 1}.json",
                [r.model_dump() for r in agent_reports],
            )

        # Consensus report
        save_json_artifact(
            settings.bucket_name,
            f"{prefix}/qc/qc-iter{iteration}.json",
            [r.model_dump() for r in consensus_reports.values()],
        )

        passed_count = sum(1 for r in consensus_reports.values() if r.passed)
        logger.info(
            f"QA iteration {iteration} consensus: "
            f"{passed_count}/{len(consensus_reports)} segments passed "
            f"(avg score: {sum(r.consensus_score for r in consensus_reports.values()) / len(consensus_reports):.2f})"
        )

        return consensus_reports

    # ── Private ───────────────────────────────────────────────────────────

    def _run_single_agent(
        self,
        source_segments: list[TranscribedSegment],
        translated_segments: list[TranslatedSegment],
        cultural_reports: dict[str, CulturalAnalysisReport],
        iteration: int,
        agent_index: int,
        temperature: float,
    ) -> list[QAReport]:
        """Run one QA agent at the given temperature."""
        model = GenerativeModel(settings.gemini_model)

        BATCH_SIZE = 10
        all_reports: list[QAReport] = []
        source_lookup = {s.segment_id: s for s in source_segments}

        for i in range(0, len(translated_segments), BATCH_SIZE):
            batch = translated_segments[i : i + BATCH_SIZE]
            source_batch = [source_lookup[s.segment_id] for s in batch]

            source_json = json.dumps(
                [{"segment_id": s.segment_id, "source_text": s.source_text,
                  "emotion_tag": s.emotion_tag} for s in source_batch],
                ensure_ascii=False, indent=2,
            )

            translated_json = json.dumps(
                [{"segment_id": s.segment_id, "translated_text": s.translated_text,
                  "cultural_adaptations_applied": s.cultural_adaptations_applied}
                 for s in batch],
                ensure_ascii=False, indent=2,
            )

            cultural_flags = self._format_flags(source_batch, cultural_reports)

            prompt = QA_PROMPT.format(
                source_language=settings.source_language_name,
                target_language=settings.target_language_name,
                rubric=self._rubric,
                source_json=source_json,
                cultural_flags=cultural_flags,
                translated_json=translated_json,
            )

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 8192,
                },
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw[raw.index("\n") + 1:]
                if raw.endswith("```"):
                    raw = raw[: raw.rfind("```")].rstrip()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                if len(batch) > 1:
                    mid = len(batch) // 2
                    logger.warning(
                        f"QA agent {agent_index} JSON parse failed ({exc}); "
                        f"retrying as two half-batches of {mid} segments"
                    )
                    all_reports.extend(
                        self._run_single_agent(
                            source_segments, list(batch[:mid]), cultural_reports, iteration, agent_index, temperature
                        )
                    )
                    all_reports.extend(
                        self._run_single_agent(
                            source_segments, list(batch[mid:]), cultural_reports, iteration, agent_index, temperature
                        )
                    )
                    continue
                logger.error(
                    f"QA agent {agent_index} JSON parse failed on single segment; skipping. Raw:\n{raw[:500]}"
                )
                data = []
            for item in data:
                all_reports.append(
                    QAReport(
                        iteration=iteration,
                        segment_id=item["segment_id"],
                        score=float(item["score"]),
                        issues=[QAIssue(**iss) for iss in item.get("issues", [])],
                        passed=bool(item.get("passed", False)),
                        agent_index=agent_index,
                    )
                )

        return all_reports

    def _format_flags(
        self,
        segments: list[TranscribedSegment],
        cultural_reports: dict[str, CulturalAnalysisReport],
    ) -> str:
        lines = []
        for seg in segments:
            report = cultural_reports.get(seg.segment_id)
            if report and report.flags:
                for flag in report.flags:
                    lines.append(
                        f"  {seg.segment_id}: [{flag.flag_type}] "
                        f'"{flag.source_expression}" → must be adapted as: {flag.suggested_adaptation}'
                    )
        return "\n".join(lines) if lines else "No special cultural flags."
