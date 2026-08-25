"""utils/segment_models.py — Shared data models for the pipeline."""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Transcription ─────────────────────────────────────────────────────────────

class TranscribedSegment(BaseModel):
    """One utterance from the source video."""
    segment_id: str                  # e.g. "seg_001"
    start_ms: int                    # milliseconds
    end_ms: int                      # milliseconds
    speaker_id: str                  # e.g. "SPEAKER_01"
    source_text: str                 # original Korean text
    emotion_tag: str = "neutral"     # neutral | excited | sad | angry | whisper


# ── Cultural intelligence ──────────────────────────────────────────────────────

class FlagSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FlagType(str, Enum):
    IDIOM = "idiom"
    HONORIFIC = "honorific"
    HUMOR = "humor"
    TABOO = "taboo"
    POP_CULTURE = "pop_culture"
    HISTORICAL = "historical"
    FOOD = "food"
    SOCIAL_NORM = "social_norm"


class CulturalFlag(BaseModel):
    flag_id: str
    segment_id: str
    source_expression: str           # the Korean phrase flagged
    flag_type: FlagType
    severity: FlagSeverity
    cultural_note: str               # why this is tricky
    suggested_adaptation: str        # recommended English equivalent


class CulturalAnalysisReport(BaseModel):
    segment_id: str
    flags: list[CulturalFlag] = Field(default_factory=list)
    has_high_severity: bool = False


# ── Translation ───────────────────────────────────────────────────────────────

class TranslatedSegment(BaseModel):
    segment_id: str
    start_ms: int
    end_ms: int
    speaker_id: str
    source_text: str
    translated_text: str
    cultural_adaptations_applied: list[str] = Field(default_factory=list)
    iteration: int = 0               # which QA iteration produced this


# ── QA ────────────────────────────────────────────────────────────────────────

class IssueSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class QAIssue(BaseModel):
    issue_id: str
    segment_id: str
    issue_type: str                  # cultural_mistranslation | mistiming | tone | line_length
    severity: IssueSeverity
    description: str
    fix_hint: str                    # concrete instruction for the translation agent


class QAReport(BaseModel):
    iteration: int
    segment_id: str
    score: float                     # 0.0–1.0
    issues: list[QAIssue] = Field(default_factory=list)
    passed: bool = False
    agent_index: Optional[int] = None   # which QA agent produced this (for consensus)


class ConsensusQAReport(BaseModel):
    """Merged report from N QA agents."""
    iteration: int
    segment_id: str
    consensus_score: float
    issues: list[QAIssue] = Field(default_factory=list)
    passed: bool = False


# ── Pipeline state ────────────────────────────────────────────────────────────

class SegmentState(BaseModel):
    """Full state for one segment through the pipeline."""
    transcribed: Optional[TranscribedSegment] = None
    cultural_report: Optional[CulturalAnalysisReport] = None
    current_translation: Optional[TranslatedSegment] = None
    qa_history: list[ConsensusQAReport] = Field(default_factory=list)
    accepted: bool = False
    final_iteration: int = 0


class PipelineState(BaseModel):
    """Top-level state for the full job."""
    job_id: str
    source_lang: str
    target_lang: str
    input_gcs_uri: str
    stem: str                        # filename without extension
    segments: dict[str, SegmentState] = Field(default_factory=dict)
    current_phase: str = "transcription"
    completed: bool = False
