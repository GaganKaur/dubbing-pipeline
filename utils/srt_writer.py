"""utils/srt_writer.py — Convert translated segments to SRT format."""
import textwrap
from utils.segment_models import TranslatedSegment


def ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp: HH:MM:SS,mmm"""
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1_000
    millis = ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def wrap_subtitle_text(text: str, max_chars: int = 42, max_lines: int = 2) -> str:
    """
    Wrap subtitle text to max_chars per line, max_lines lines.
    If text still exceeds limits after wrapping, truncate with ellipsis.
    """
    # Hard wrap at max_chars
    lines = textwrap.wrap(text, width=max_chars)

    if len(lines) > max_lines:
        # Truncate — keep first max_lines lines, add ellipsis to last
        lines = lines[:max_lines]
        if len(lines[-1]) <= max_chars - 3:
            lines[-1] = lines[-1] + "..."
        else:
            lines[-1] = lines[-1][: max_chars - 3] + "..."

    return "\n".join(lines)


def segments_to_srt(
    segments: list[TranslatedSegment],
    max_chars_per_line: int = 42,
    max_lines_per_cue: int = 2,
) -> str:
    """Convert a list of TranslatedSegment to a complete SRT string."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = ms_to_srt_time(seg.start_ms)
        end = ms_to_srt_time(seg.end_ms)
        text = wrap_subtitle_text(
            seg.translated_text,
            max_chars=max_chars_per_line,
            max_lines=max_lines_per_cue,
        )
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(lines)


def segments_to_webvtt(segments: list[TranslatedSegment]) -> str:
    """Convert segments to WebVTT format (for web/streaming players)."""
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(segments, start=1):
        start = ms_to_srt_time(seg.start_ms).replace(",", ".")
        end = ms_to_srt_time(seg.end_ms).replace(",", ".")
        text = wrap_subtitle_text(seg.translated_text)
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)
