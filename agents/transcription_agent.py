"""agents/transcription_agent.py — Extract timed Korean speech segments from video.

Two-pass design:
  1. Cloud Speech-to-Text (Chirp) measures real, audio-anchored word timestamps
     and groups them into subtitle-sized cues (tools/speech_to_text_tool.py).
  2. Gemini enriches each cue with speaker_id, emotion_tag, and (only if the
     speech-to-text system clearly erred) a corrected source_text — but cannot
     add, remove, merge, split, or re-time segments.

Splitting the work this way means timing comes from a measurement, not an
LLM estimate, while Gemini still drives everything text/cultural downstream.
"""
from __future__ import annotations
import json
import logging
import os
import subprocess
import tempfile

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

from config.settings import settings
from tools.gcs_tool import download_file, gcs_uri_to_parts, save_json_artifact, load_checkpoint
from tools.speech_to_text_tool import transcribe_with_timestamps
from utils.segment_models import TranscribedSegment

logger = logging.getLogger(__name__)

ENRICHMENT_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "segment_id":  {"type": "string"},
            "source_text": {"type": "string"},
            "speaker_id":  {"type": "string"},
            "emotion_tag": {
                "type": "string",
                "enum": ["neutral","excited","sad","angry","whisper","laughing"],
            },
        },
        "required": ["segment_id","source_text","speaker_id","emotion_tag"],
    },
}

ENRICHMENT_PROMPT = """A speech-to-text system already transcribed this audio clip into timed segments.
Your job is NOT to re-transcribe or re-time anything — only to enrich each segment.

Listen to the audio and, for EVERY segment below, return:
- segment_id: copy exactly as given
- source_text: the Korean text, copied as-is UNLESS the speech-to-text system clearly mis-transcribed a word (check against what you hear) — in that case, correct only that word
- speaker_id: SPEAKER_01, SPEAKER_02 etc — be consistent for the same voice across segments
- emotion_tag: neutral | excited | sad | angry | whisper | laughing — based on tone of voice

Return exactly {n} objects, one per input segment, in the same order, with matching segment_id.
Do not add, remove, merge, or split segments.

Input segments:
{segments_json}

Return only the JSON array."""


class TranscriptionAgent:
    def __init__(self):
        vertexai.init(project=settings.project_id, location=settings.gcp_region)
        self.model = GenerativeModel(settings.gemini_model)

    def run(self, video_gcs_uri: str, stem: str, resume: bool = False) -> list[TranscribedSegment]:
        prefix = settings.source_prefix(stem)
        cached = load_checkpoint(settings.bucket_name, prefix, "trans_segments", 0, resume)
        if cached:
            logger.info(f"Loaded {len(cached)} segments from checkpoint")
            return [TranscribedSegment(**s) for s in cached]

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Downloading video: {video_gcs_uri}")
            bucket, path = gcs_uri_to_parts(video_gcs_uri)
            local_video = os.path.join(tmpdir, "input.mp4")
            download_file(bucket, path, local_video)

            logger.info("Extracting audio track with FFmpeg...")
            local_audio = os.path.join(tmpdir, "audio.wav")
            self._extract_audio(local_video, local_audio)

            duration_s = self._get_duration(local_audio)
            chunk_duration_s = settings.stt_chunk_duration_s
            logger.info(f"Audio duration: {duration_s:.1f}s — splitting into {chunk_duration_s}s chunks")

            chunk_paths: list[str] = []
            chunk_offsets_ms: list[int] = []
            offset_s = 0.0
            chunk_num = 0

            while offset_s < duration_s:
                chunk_num += 1
                chunk_path = os.path.join(tmpdir, f"chunk_{chunk_num:03d}.wav")
                self._extract_chunk(local_audio, chunk_path, offset_s, chunk_duration_s)
                chunk_paths.append(chunk_path)
                chunk_offsets_ms.append(int(offset_s * 1000))
                offset_s += chunk_duration_s

            logger.info(f"Running Chirp forced alignment on {chunk_num} chunk(s)...")
            raw_segments = transcribe_with_timestamps(chunk_paths, chunk_offsets_ms)
            logger.info(f"Chirp produced {len(raw_segments)} timed segments")

            for i, seg in enumerate(raw_segments, start=1):
                seg["segment_id"] = f"seg_{i:03d}"

            # Enrich each chunk's segments with Gemini (speaker_id, emotion_tag, text check)
            all_segments: list[TranscribedSegment] = []
            chunk_duration_ms = chunk_duration_s * 1000

            for chunk_path, offset_ms in zip(chunk_paths, chunk_offsets_ms):
                chunk_end_ms = offset_ms + chunk_duration_ms
                chunk_raw = [s for s in raw_segments if offset_ms <= s["start_ms"] < chunk_end_ms]
                if not chunk_raw:
                    continue

                chunk_asr = [s for s in chunk_raw if not s.get("manual")]
                chunk_manual = [s for s in chunk_raw if s.get("manual")]

                if chunk_asr:
                    logger.info(f"Enriching {len(chunk_asr)} segments for chunk at {offset_ms}ms...")
                    all_segments.extend(self._enrich_chunk(chunk_path, chunk_asr))

                for s in chunk_manual:
                    logger.info(f"Using human-confirmed text for segment at {s['start_ms']}ms: {s['source_text']!r}")
                    all_segments.append(TranscribedSegment(
                        segment_id=s["segment_id"],
                        start_ms=s["start_ms"],
                        end_ms=s["end_ms"],
                        speaker_id=s["speaker_id"],
                        source_text=s["source_text"],
                        emotion_tag=s["emotion_tag"],
                    ))

        # Sort by start_ms and re-assign clean sequential IDs
        all_segments.sort(key=lambda s: s.start_ms)
        for i, seg in enumerate(all_segments, start=1):
            seg.segment_id = f"seg_{i:03d}"

        save_json_artifact(
            settings.bucket_name,
            f"{prefix}/trans_segments/trans_segments-iter0.json",
            [s.model_dump() for s in all_segments],
        )
        logger.info(f"Transcription complete: {len(all_segments)} segments across {chunk_num} chunks")
        return all_segments

    # ── Private ───────────────────────────────────────────────────────────

    def _extract_audio(self, video_path: str, audio_path: str) -> None:
        cmd = [
            settings.ffmpeg_bin, "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extract failed: {result.stderr[-300:]}")

    def _extract_chunk(self, audio_path: str, chunk_path: str, offset_s: float, duration_s: float) -> None:
        cmd = [
            settings.ffmpeg_bin, "-y",
            "-ss", str(offset_s),
            "-t", str(duration_s),
            "-i", audio_path,
            chunk_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg chunk extract failed: {result.stderr[-200:]}")

    def _get_duration(self, audio_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def _enrich_chunk(self, chunk_path: str, raw_segments: list[dict]) -> list[TranscribedSegment]:
        with open(chunk_path, "rb") as f:
            audio_bytes = f.read()

        segments_json = json.dumps(
            [{"segment_id": s["segment_id"], "source_text": s["source_text"]} for s in raw_segments],
            ensure_ascii=False,
        )
        prompt = ENRICHMENT_PROMPT.format(n=len(raw_segments), segments_json=segments_json)

        enriched_by_id: dict[str, dict] = {}
        try:
            response = self.model.generate_content(
                [Part.from_data(audio_bytes, mime_type="audio/wav"), prompt],
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    response_schema=ENRICHMENT_RESPONSE_SCHEMA,
                ),
            )
            for item in json.loads(response.text.strip()):
                enriched_by_id[item["segment_id"]] = item
        except Exception as e:
            logger.warning(f"Enrichment failed for chunk: {e} — using defaults")

        result = []
        for s in raw_segments:
            e = enriched_by_id.get(s["segment_id"], {})
            result.append(TranscribedSegment(
                segment_id=s["segment_id"],
                start_ms=s["start_ms"],
                end_ms=s["end_ms"],
                speaker_id=e.get("speaker_id", "SPEAKER_01"),
                source_text=e.get("source_text", s["source_text"]),
                emotion_tag=e.get("emotion_tag", "neutral"),
            ))
        return result
