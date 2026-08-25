"""main.py — ADK pipeline orchestrator. Runs locally and inside Cloud Run Jobs."""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

from config.settings import settings
from agents.transcription_agent import TranscriptionAgent
from agents.cultural_intel_agent import CulturalIntelAgent
from agents.translation_agent import TranslationAgent
from agents.qa_agent import QAAgent
from tools.subtitle_tool import SubtitleTool
from tools.gcs_tool import save_json_artifact, upload_string
from utils.segment_models import TranslatedSegment, PipelineState


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline.orchestrator")


def run_pipeline(
    input_gcs_uri: str,
    source_lang: str = "ko",
    target_lang: str = "en",
    job_id: str = "local-dev",
    max_iterations: int = None,
    resume: bool = False,
) -> dict:
    """
    Full pipeline:
      1. Transcription
      2. Cultural intelligence
      3. Translation + QA loop (up to max_iterations)
      4. Subtitle generation + burning

    Returns summary dict with output GCS URIs.
    """
    max_iterations = max_iterations or settings.max_iterations
    stem = Path(input_gcs_uri).stem  # e.g. "trailer" from gs://.../trailer.mp4

    # Keep the settings singleton in sync with this run's languages — agents
    # and output-path helpers read settings.source_lang/target_lang directly.
    settings.source_lang = source_lang
    settings.target_lang = target_lang

    logger.info(
        f"=== Pipeline start =========================\n"
        f"  Job:    {job_id}\n"
        f"  Input:  {input_gcs_uri}\n"
        f"  Langs:  {source_lang} → {target_lang}\n"
        f"  MaxIter:{max_iterations}\n"
        f"  Resume: {resume}\n"
        f"==========================================="
    )

    # ── Stage 1: Transcription ────────────────────────────────────────────
    logger.info("── Stage 1: Transcription ──────────────────")
    transcription_agent = TranscriptionAgent()
    segments = transcription_agent.run(
        video_gcs_uri=input_gcs_uri,
        stem=stem,
        resume=resume,
    )
    logger.info(f"Transcribed {len(segments)} segments")

    # ── Stage 2: Cultural intelligence ───────────────────────────────────
    logger.info("── Stage 2: Cultural intelligence ──────────")
    cultural_agent = CulturalIntelAgent()
    cultural_reports = cultural_agent.run(
        segments=segments,
        stem=stem,
        resume=resume,
    )
    total_flags = sum(len(r.flags) for r in cultural_reports.values())
    high_flags = sum(1 for r in cultural_reports.values() if r.has_high_severity)
    logger.info(f"Cultural analysis: {total_flags} flags ({high_flags} high-severity)")

    # ── Stage 3: Translation + QA loop ───────────────────────────────────
    logger.info("── Stage 3: Translation + QA loop ──────────")
    translation_agent = TranslationAgent()
    qa_agent = QAAgent()

    # Track which segments have passed QA — start with all pending
    qa_critiques = None
    final_translations: dict[str, TranslatedSegment] = {}

    for iteration in range(max_iterations):
        logger.info(f"── Iteration {iteration + 1}/{max_iterations} ──────────")

        # Translate (first pass: all segments; subsequent: only failed ones)
        translations = translation_agent.run(
            segments=segments,
            cultural_reports=cultural_reports,
            stem=stem,
            iteration=iteration,
            qa_critiques=qa_critiques,
        )

        # Merge new translations into final dict (preserve passed ones)
        for t in translations:
            final_translations[t.segment_id] = t

        # Run QA evaluation
        qa_results = qa_agent.run(
            source_segments=segments,
            translated_segments=list(final_translations.values()),
            cultural_reports=cultural_reports,
            stem=stem,
            iteration=iteration + 1,
        )

        # Count results
        passed = sum(1 for r in qa_results.values() if r.passed)
        total = len(qa_results)
        avg_score = sum(r.consensus_score for r in qa_results.values()) / total
        logger.info(
            f"QA iteration {iteration + 1}: "
            f"{passed}/{total} passed | avg score: {avg_score:.3f}"
        )

        # Check if all segments passed
        all_passed = passed == total
        last_iteration = iteration == max_iterations - 1

        if all_passed:
            logger.info(f"All segments passed QA at iteration {iteration + 1}!")
            break

        if last_iteration:
            logger.info(
                f"Max iterations reached. Accepting best-effort results. "
                f"{total - passed} segments below threshold."
            )
            break

        # Prepare critiques for next iteration (only failed segments)
        qa_critiques = {
            seg_id: report
            for seg_id, report in qa_results.items()
            if not report.passed
        }

    # ── Build final ordered translation list ─────────────────────────────
    # Sort by start_ms to preserve video order
    seg_order = {s.segment_id: s.start_ms for s in segments}
    sorted_translations = sorted(
        final_translations.values(),
        key=lambda t: seg_order.get(t.segment_id, 0),
    )

    # ── Stage 4: Subtitle generation ─────────────────────────────────────
    logger.info("── Stage 4: Subtitle generation ─────────────")
    subtitle_tool = SubtitleTool(
        bucket_name=settings.bucket_name,
        max_chars_per_line=settings.max_chars_per_line,
        max_lines_per_cue=settings.max_lines_per_cue,
    )

    srt_gcs_path = settings.srt_output_path(stem)
    srt_uri = subtitle_tool.generate_srt(
        segments=sorted_translations,
        output_gcs_path=srt_gcs_path,
    )
    logger.info(f"SRT generated: {srt_uri}")

    # Also generate WebVTT for web players
    vtt_uri = subtitle_tool.generate_webvtt(
        segments=sorted_translations,
        output_gcs_path=srt_gcs_path.replace(".srt", ".vtt"),
    )

    outputs = {
        "srt_uri": srt_uri,
        "vtt_uri": vtt_uri,
        "segments_count": len(sorted_translations),
    }

    # ── Stage 4b: Subtitle burning (optional) ─────────────────────────────
    if settings.burn_subtitles:
        logger.info("── Stage 4b: Burning subtitles into video ──")
        try:
            burned_uri = subtitle_tool.burn_subtitles(
                video_gcs_uri=input_gcs_uri,
                srt_gcs_path=srt_gcs_path,
                output_gcs_path=settings.final_output_path(stem),
            )
            outputs["burned_video_uri"] = burned_uri
            logger.info(f"Burned video: {burned_uri}")
        except Exception as e:
            logger.error(f"Subtitle burning failed: {e} — SRT still available")
            outputs["burn_error"] = str(e)

    if settings.soft_subtitles:
        logger.info("── Stage 4c: Muxing soft subtitle track ────")
        try:
            soft_uri = subtitle_tool.mux_soft_subtitles(
                video_gcs_uri=input_gcs_uri,
                srt_gcs_path=srt_gcs_path,
                output_gcs_path=settings.final_output_path(stem).replace(
                    "_dubbed.mp4", "_soft_subs.mp4"
                ),
                language_code=target_lang,
            )
            outputs["soft_subtitled_uri"] = soft_uri
            logger.info(f"Soft subtitled video: {soft_uri}")
        except Exception as e:
            logger.error(f"Soft subtitle mux failed: {e}")

    # ── Save pipeline summary ─────────────────────────────────────────────
    summary = {
        "job_id": job_id,
        "input": input_gcs_uri,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "segments": len(sorted_translations),
        "cultural_flags": total_flags,
        "iterations_run": iteration + 1,
        "outputs": outputs,
    }
    save_json_artifact(
        settings.bucket_name,
        f"{settings.output_prefix(stem)}/pipeline_summary.json",
        summary,
    )

    logger.info(
        f"=== Pipeline complete =====================\n"
        f"  Segments:   {len(sorted_translations)}\n"
        f"  Flags:      {total_flags}\n"
        f"  Iterations: {iteration + 1}\n"
        f"  SRT:        {srt_uri}\n"
        f"==========================================="
    )
    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Korean → English dubbing pipeline")
    parser.add_argument("--input", required=True, help="gs:// URI of source video")
    parser.add_argument("--source-lang", default="ko")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--job-id", default="local-dev")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--build-rag",
        action="store_true",
        help="Run RAG builder before pipeline (one-time setup)",
    )
    args = parser.parse_args()

    if args.build_rag:
        from agents.rag_builder_agent import RAGBuilderAgent
        logger.info("Building cultural RAG knowledge base...")
        rag_agent = RAGBuilderAgent()
        count = rag_agent.run(num_passes=3)
        logger.info(f"RAG built: {count} entries indexed")

    summary = run_pipeline(
        input_gcs_uri=args.input,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        job_id=args.job_id,
        max_iterations=args.max_iterations,
        resume=args.resume,
    )
    print("\nOutputs:")
    for key, val in summary["outputs"].items():
        print(f"  {key}: {val}")
