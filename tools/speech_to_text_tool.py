"""tools/speech_to_text_tool.py — Forced-alignment timestamps via Cloud Speech-to-Text (Chirp).

Cloud Speech-to-Text measures where each word actually falls in the audio
(word-level timestamps), unlike an LLM holistically estimating timestamps.
This module owns TIMING ONLY — text/cultural enrichment is handled by Gemini
in agents/transcription_agent.py.

Three-tier design:
  1. Primary (chirp_2, us-central1) — forced alignment for the whole clip.
     Its segments are authoritative: timing and text are never altered.
  2. Secondary (chirp_3, "us") — a second, independent recognition pass over
     the same audio, used ONLY to recover words that fall in gaps the
     primary pass left (e.g. a line buried under heavy background music).
     If the secondary pass fails or finds nothing new, behavior is identical
     to the primary pass alone.
  3. Manual corrections (config/manual_corrections.json) — for the residual
     gaps neither ASR model can recover, a human reviews the audio once and
     records the correct text. Only used to fill gaps still left after tiers
     1-2; never overrides an ASR segment.

Tiers 1-2 are deterministic ASR measurements — no generative "guessing", so
no hallucination risk (see TECHNICAL.md for the rejected Gemini-based
gap-fill approach). Tier 3 is human-confirmed, not generated.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from config.settings import settings

logger = logging.getLogger(__name__)

MANUAL_CORRECTIONS_PATH = Path(__file__).parent.parent / "config" / "manual_corrections.json"

# Start a new subtitle cue after this much silence since the previous word.
GAP_THRESHOLD_MS = 700
# Force a new cue if the current one would otherwise grow longer than this.
MAX_SEGMENT_MS = 7000
# Short interjections ("아!") get a real but tiny duration from Chirp —
# stretch them so they're actually readable as a subtitle cue.
MIN_SEGMENT_MS = 800
# A secondary-pass word within this many ms of a primary segment's edge is
# treated as the same utterance (already covered), not a new one.
COVERAGE_BUFFER_MS = 300

_clients: dict[str, SpeechClient] = {}


def _get_client(region: str) -> SpeechClient:
    if region not in _clients:
        _clients[region] = SpeechClient(client_options={"api_endpoint": f"{region}-speech.googleapis.com"})
    return _clients[region]


def _recognize_words(chunk_paths: list[str], chunk_offsets_ms: list[int], model: str, region: str) -> list[dict]:
    """Run Speech-to-Text on each chunk and return a flat, time-sorted word list."""
    client = _get_client(region)
    recognizer = f"projects/{settings.project_id}/locations/{region}/recognizers/_"

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[settings.stt_language_code],
        model=model,
        features=cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        ),
    )

    words: list[dict] = []
    for chunk_path, offset_ms in zip(chunk_paths, chunk_offsets_ms):
        with open(chunk_path, "rb") as f:
            audio_bytes = f.read()
        if len(audio_bytes) < 1000:
            continue

        request = cloud_speech.RecognizeRequest(recognizer=recognizer, config=config, content=audio_bytes)
        try:
            response = client.recognize(request=request)
        except Exception as e:
            logger.warning(f"Speech-to-Text ({model}) failed for chunk at offset {offset_ms}ms: {e} — skipping chunk")
            continue

        for result in response.results:
            if not result.alternatives:
                continue
            for word in result.alternatives[0].words:
                words.append({
                    "text": word.word,
                    "start_ms": offset_ms + int(word.start_offset.total_seconds() * 1000),
                    "end_ms": offset_ms + int(word.end_offset.total_seconds() * 1000),
                })

    words.sort(key=lambda w: w["start_ms"])
    return words


def transcribe_with_timestamps(
    chunk_paths: list[str],
    chunk_offsets_ms: list[int],
) -> list[dict]:
    """Transcribe audio chunks and return word-anchored subtitle segments.

    Each chunk is recognized independently (sync `recognize` caps at ~60s of
    audio); per-word timestamps are real measurements, so adding the chunk's
    offset back in introduces no drift or chunk-boundary bias.

    Returns a list of {"start_ms", "end_ms", "source_text"} dicts, sorted by
    start_ms, covering the whole clip.
    """
    primary_words = _recognize_words(chunk_paths, chunk_offsets_ms, settings.stt_model, settings.gcp_region)
    primary_segments = _words_to_segments(primary_words)

    recovered_segments: list[dict] = []
    try:
        fallback_words = _recognize_words(
            chunk_paths, chunk_offsets_ms, settings.stt_fallback_model, settings.stt_fallback_region
        )
        uncovered = [w for w in fallback_words if not _covered_by_any(w, primary_segments)]
        recovered_segments = _words_to_segments(uncovered)
        if recovered_segments:
            logger.info(f"Secondary pass ({settings.stt_fallback_model}) recovered {len(recovered_segments)} segment(s)")
    except Exception as e:
        logger.warning(f"Secondary STT pass ({settings.stt_fallback_model}) failed: {e} — continuing with primary only")

    covered = primary_segments + recovered_segments
    manual_segments = [c for c in _load_manual_corrections() if not _overlaps_any(c, covered)]
    if manual_segments:
        logger.info(f"Applied {len(manual_segments)} human-confirmed correction(s) for residual gap(s)")

    return _merge_segments(primary_segments, recovered_segments + manual_segments)


def _load_manual_corrections() -> list[dict]:
    """Human-confirmed segments for gaps neither ASR pass can recover (tier 3)."""
    if not MANUAL_CORRECTIONS_PATH.exists():
        return []
    with open(MANUAL_CORRECTIONS_PATH) as f:
        corrections = json.load(f)
    return [
        {
            "start_ms": c["start_ms"],
            "end_ms": c["end_ms"],
            "source_text": c["source_text"],
            "speaker_id": c.get("speaker_id", "SPEAKER_01"),
            "emotion_tag": c.get("emotion_tag", "neutral"),
            "manual": True,
        }
        for c in corrections
    ]


def _covered_by_any(word: dict, segments: list[dict], buffer_ms: int = COVERAGE_BUFFER_MS) -> bool:
    return any(
        word["start_ms"] < seg["end_ms"] + buffer_ms and word["end_ms"] > seg["start_ms"] - buffer_ms
        for seg in segments
    )


def _overlaps_any(segment: dict, segments: list[dict]) -> bool:
    """Strict time-range overlap (no buffer) — used for manual corrections,
    which are full multi-second segments, unlike the ~300ms words
    _covered_by_any is designed for. Segments that merely touch at an edge
    (e.g. a correction filling the exact gap between two segments) are not
    considered overlapping."""
    return any(
        segment["start_ms"] < seg["end_ms"] and segment["end_ms"] > seg["start_ms"]
        for seg in segments
    )


def _merge_segments(primary: list[dict], recovered: list[dict]) -> list[dict]:
    """Combine primary (authoritative) and recovered segments, sorted by
    start_ms, trimming any recovered segment that would overlap a primary
    one once stretched to MIN_SEGMENT_MS."""
    merged = sorted(primary + recovered, key=lambda s: s["start_ms"])
    for i in range(len(merged) - 1):
        if merged[i]["end_ms"] > merged[i + 1]["start_ms"]:
            merged[i]["end_ms"] = merged[i + 1]["start_ms"]
    return [s for s in merged if s["end_ms"] > s["start_ms"]]


def _words_to_segments(words: list[dict]) -> list[dict]:
    """Group a flat word list into subtitle-sized cues using natural pauses."""
    if not words:
        return []

    words.sort(key=lambda w: w["start_ms"])

    segments: list[dict] = []
    current = [words[0]]

    for word in words[1:]:
        gap = word["start_ms"] - current[-1]["end_ms"]
        duration_if_added = word["end_ms"] - current[0]["start_ms"]

        if gap > GAP_THRESHOLD_MS or duration_if_added > MAX_SEGMENT_MS:
            segments.append(_finalize_segment(current))
            current = [word]
        else:
            current.append(word)

    segments.append(_finalize_segment(current))
    return segments


def _finalize_segment(words: list[dict]) -> dict:
    start_ms = words[0]["start_ms"]
    end_ms = max(words[-1]["end_ms"], start_ms + MIN_SEGMENT_MS)
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source_text": " ".join(w["text"] for w in words),
    }
