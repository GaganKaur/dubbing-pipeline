
"""config/settings.py — All env vars in one place. Import this everywhere."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


_LANGUAGE_NAMES = {
    "en": "English", "ko": "Korean", "es": "Spanish", "fr": "French",
    "de": "German", "ja": "Japanese", "zh": "Chinese", "pt": "Portuguese",
    "it": "Italian", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
}


@dataclass
class Settings:
    # ── GCP ──────────────────────────────────────────────────────────────
    project_id: str = field(default_factory=lambda: os.environ["PROJECT_ID"])
    gcp_region: str = field(default_factory=lambda: os.getenv("GCP_REGION", "us-central1"))
    bucket_name: str = field(default_factory=lambda: os.environ["BUCKET_NAME"])

    # ── Gemini model ─────────────────────────────────────────────────────
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    )

    # ── Pipeline ─────────────────────────────────────────────────────────
    source_lang: str = field(default_factory=lambda: os.getenv("SOURCE_LANG", "ko"))
    target_lang: str = field(default_factory=lambda: os.getenv("TARGET_LANG", "en"))
    job_id: str = field(default_factory=lambda: os.getenv("JOB_ID", "local-dev"))
    input_file: str = field(default_factory=lambda: os.getenv("INPUT_FILE", ""))

    # ── QA loop ──────────────────────────────────────────────────────────
    max_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "3"))
    )
    qa_threshold: float = field(
        default_factory=lambda: float(os.getenv("QA_THRESHOLD", "0.80"))
    )
    num_qa_agents: int = field(
        default_factory=lambda: int(os.getenv("NUM_QA_AGENTS", "3"))
    )
    qa_temperatures: list = field(
        default_factory=lambda: [
            float(t)
            for t in os.getenv("QA_TEMPERATURES", "0.1,0.5,0.9").split(",")
        ]
    )

    # ── FFmpeg ───────────────────────────────────────────────────────────
    ffmpeg_bin: str = field(default_factory=lambda: os.getenv("FFMPEG_BIN", "ffmpeg"))

    # ── Speech-to-Text (Chirp) ───────────────────────────────────────────
    # Cloud Speech-to-Text v2 forced-alignment model — provides real,
    # audio-anchored timestamps (Gemini handles text/cultural enrichment only).
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "chirp_2"))
    stt_language_code: str = field(
        default_factory=lambda: os.getenv(
            "STT_LANGUAGE_CODE",
            {"ko": "ko-KR", "en": "en-US", "ja": "ja-JP", "zh": "zh-CN",
             "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "pt": "pt-PT",
             "it": "it-IT", "ru": "ru-RU", "ar": "ar-SA", "hi": "hi-IN",
             }.get(os.getenv("SOURCE_LANG", "ko"), "ko-KR"),
        )
    )
    # Sync Recognize caps at 60s of audio; 50s leaves headroom.
    stt_chunk_duration_s: int = field(
        default_factory=lambda: int(os.getenv("STT_CHUNK_DURATION_S", "50"))
    )
    # Secondary STT pass used only to recover words chirp_2 missed under
    # heavy background music (see TECHNICAL.md). chirp_2 and chirp_3 are
    # each available in different regions, so this has its own region.
    stt_fallback_model: str = field(
        default_factory=lambda: os.getenv("STT_FALLBACK_MODEL", "chirp_3")
    )
    stt_fallback_region: str = field(
        default_factory=lambda: os.getenv("STT_FALLBACK_REGION", "us")
    )

    # ── Subtitle burning ─────────────────────────────────────────────────
    max_chars_per_line: int = field(
        default_factory=lambda: int(os.getenv("MAX_CHARS_PER_LINE", "42"))
    )
    max_lines_per_cue: int = field(
        default_factory=lambda: int(os.getenv("MAX_LINES_PER_CUE", "2"))
    )
    burn_subtitles: bool = field(
        default_factory=lambda: os.getenv("BURN_SUBTITLES", "true").lower() == "true"
    )
    soft_subtitles: bool = field(
        default_factory=lambda: os.getenv("SOFT_SUBTITLES", "true").lower() == "true"
    )

    # ── Vector Search (RAG) ──────────────────────────────────────────────
    vector_search_index_id: str = field(
        default_factory=lambda: os.getenv("VECTOR_SEARCH_INDEX_ID", "")
    )
    vector_search_endpoint_id: str = field(
        default_factory=lambda: os.getenv("VECTOR_SEARCH_ENDPOINT_ID", "")
    )
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "10")))

    # ── Checkpointing ────────────────────────────────────────────────────
    resume_from_checkpoints: bool = field(
        default_factory=lambda: os.getenv("RESUME_FROM_CHECKPOINTS", "false").lower()
        == "true"
    )

    # ── Batch regions (quota rotation) ───────────────────────────────────
    batch_regions: list = field(
        default_factory=lambda: os.getenv(
            "BATCH_REGIONS", "us-central1"
        ).split(",")
    )

    def output_prefix(self, stem: str) -> str:
        """GCS path prefix for all artifacts for this job."""
        return f"output/artifacts/{self.target_lang}/{stem}"

    @property
    def source_language_name(self) -> str:
        """Full English name of source_lang, e.g. 'ko' -> 'Korean'."""
        return _LANGUAGE_NAMES.get(self.source_lang, self.source_lang.upper())

    @property
    def target_language_name(self) -> str:
        """Full English name of target_lang, e.g. 'es' -> 'Spanish'."""
        return _LANGUAGE_NAMES.get(self.target_lang, self.target_lang.upper())

    def source_prefix(self, stem: str) -> str:
        """GCS path prefix for source-language artifacts (transcription,
        cultural analysis) — keyed by source_lang so Korean and Chinese
        source artifacts for the same video never collide, but shared
        across all target languages for the same source."""
        return f"output/artifacts/_source/{self.source_lang}/{stem}"

    def final_output_path(self, stem: str) -> str:
        return f"output/{self.target_lang}/{stem}_{self.target_lang}_dubbed.mp4"

    def srt_output_path(self, stem: str) -> str:
        return f"output/{self.target_lang}/{stem}_{self.target_lang}.srt"


# Singleton — import and use everywhere
settings = Settings()
