"""tools/subtitle_tool.py — Generate SRT files and burn subtitles into video."""
from __future__ import annotations
import logging
import os
import re
import subprocess
import tempfile

from config.settings import settings
from utils.segment_models import TranslatedSegment
from utils.srt_writer import segments_to_srt, segments_to_webvtt
from tools.gcs_tool import upload_file, upload_string, download_file, gcs_uri_to_parts

logger = logging.getLogger(__name__)

# QuickTime/MOV subtitle tracks require ISO 639-2 (3-letter) language codes.
# ISO 639-1 (2-letter) codes are silently dropped by ffmpeg's mov muxer,
# which leaves the track without a language tag and hidden from QuickTime's
# subtitle menu.
_ISO_639_1_TO_2 = {
    "en": "eng", "ko": "kor", "ja": "jpn", "zh": "zho",
    "es": "spa", "fr": "fra", "de": "deu", "pt": "por",
    "it": "ita", "ru": "rus", "ar": "ara", "hi": "hin",
}


class SubtitleTool:
    """
    Handles SRT generation and FFmpeg subtitle burning.

    Two modes:
      - soft_only:  mux SRT as a subtitle track (no re-encode, fast)
      - burned_in:  render subtitles into video frames (re-encode, slower)

    For the prototype we do both: soft track for streaming players,
    burned-in copy for review and demo.
    """

    def __init__(
        self,
        bucket_name: str,
        max_chars_per_line: int = 42,
        max_lines_per_cue: int = 2,
        font_size: int = 24,
        font_color: str = "white",
        outline_color: str = "black",
        outline_width: int = 2,
    ):
        self.bucket_name = bucket_name
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_cue = max_lines_per_cue
        self.font_size = font_size
        self.font_color = font_color
        self.outline_color = outline_color
        self.outline_width = outline_width

    # ── Public API ────────────────────────────────────────────────────────

    def generate_srt(
        self,
        segments: list[TranslatedSegment],
        output_gcs_path: str,
    ) -> str:
        """Convert segments to SRT and upload to GCS. Returns gs:// URI."""
        srt_content = segments_to_srt(
            segments,
            max_chars_per_line=self.max_chars_per_line,
            max_lines_per_cue=self.max_lines_per_cue,
        )
        uri = upload_string(self.bucket_name, output_gcs_path, srt_content, "text/plain")
        logger.info(f"SRT generated: {uri} ({len(segments)} cues)")
        return uri

    def generate_webvtt(
        self,
        segments: list[TranslatedSegment],
        output_gcs_path: str,
    ) -> str:
        """Convert segments to WebVTT and upload to GCS."""
        vtt_content = segments_to_webvtt(segments)
        return upload_string(self.bucket_name, output_gcs_path, vtt_content, "text/vtt")

    @staticmethod
    def _check_libass() -> None:
        """Raise a clear error if FFmpeg was compiled without libass."""
        result = subprocess.run(
            [settings.ffmpeg_bin, "-filters"],
            capture_output=True, text=True,
        )
        if "ass" not in result.stdout and "subtitles" not in result.stdout:
            raise RuntimeError(
                "FFmpeg is missing libass support — subtitle burning is unavailable.\n"
                "Fix: brew install ffmpeg-full and set FFMPEG_BIN to its path\n"
                "(Soft subtitle track was still created successfully.)"
            )

    def burn_subtitles(
        self,
        video_gcs_uri: str,
        srt_gcs_path: str,
        output_gcs_path: str,
    ) -> str:
        """
        Download video + SRT from GCS, burn subs with FFmpeg, upload result.
        Returns output gs:// URI.
        """
        self._check_libass()
        with tempfile.TemporaryDirectory() as tmpdir:
            video_bucket, video_path = gcs_uri_to_parts(video_gcs_uri)
            local_video = os.path.join(tmpdir, "input.mp4")
            local_srt = os.path.join(tmpdir, "subtitles.srt")
            local_ass = os.path.join(tmpdir, "subtitles.ass")
            local_output = os.path.join(tmpdir, "output_burned.mp4")

            logger.info("Downloading video from GCS...")
            download_file(video_bucket, video_path, local_video)

            logger.info("Downloading SRT from GCS...")
            download_file(self.bucket_name, srt_gcs_path, local_srt)

            # Convert SRT → ASS so styling is baked into the file header.
            # This avoids FFmpeg 8's filter-option comma-parsing issues entirely.
            self._srt_to_styled_ass(local_srt, local_ass)

            # FFmpeg 8 removed implicit positional option passing — the filename
            # must be supplied as an explicit named option: ass=filename=...
            escaped_ass = local_ass.replace("\\", "\\\\").replace(":", "\\:")
            cmd = [
                settings.ffmpeg_bin, "-y",
                "-i", local_video,
                "-vf", f"ass=filename={escaped_ass}",
                "-c:v", "libx264",
                "-crf", "23",
                "-preset", "fast",
                "-c:a", "copy",
                local_output,
            ]

            logger.info(f"Running FFmpeg burn: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed (code {result.returncode}): {result.stderr[-500:]}")

            logger.info("FFmpeg complete. Uploading output...")
            return upload_file(self.bucket_name, output_gcs_path, local_output)

    # ── Private ───────────────────────────────────────────────────────────

    def _srt_to_styled_ass(self, local_srt: str, local_ass: str) -> None:
        """
        Convert an SRT file to ASS and patch the default Style line so our
        font size and outline settings are baked directly into the file.
        The ass= filter then renders it without needing any force_style option.
        """
        result = subprocess.run(
            [settings.ffmpeg_bin, "-y", "-i", local_srt, local_ass],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"SRT→ASS conversion failed: {result.stderr[-300:]}")

        with open(local_ass, encoding="utf-8") as f:
            content = f.read()

        # ASS Style columns: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,
        #   OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,
        #   ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,
        #   Alignment,MarginL,MarginR,MarginV,Encoding
        styled = (
            f"Style: Default,Arial,{self.font_size},"
            f"&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            f"0,0,0,0,100,100,0,0,1,{self.outline_width},0,2,10,10,10,1"
        )
        content = re.sub(r"^Style: Default,.*$", styled, content, flags=re.MULTILINE)

        with open(local_ass, "w", encoding="utf-8") as f:
            f.write(content)

    def mux_soft_subtitles(
        self,
        video_gcs_uri: str,
        srt_gcs_path: str,
        output_gcs_path: str,
        language_code: str = "en",
    ) -> str:
        """
        Mux SRT as a soft subtitle track into MP4 (no re-encode).
        Much faster than burning — stream copy for both video and audio.
        Returns output gs:// URI.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            video_bucket, video_path = gcs_uri_to_parts(video_gcs_uri)

            # Match the output container to the input to avoid codec-container
            # mismatches (e.g. ProRes cannot be stream-copied into MP4).
            input_ext = os.path.splitext(video_path)[1].lower() or ".mp4"
            local_video = os.path.join(tmpdir, f"input{input_ext}")
            local_srt = os.path.join(tmpdir, "subtitles.srt")
            local_output = os.path.join(tmpdir, f"output_soft{input_ext}")

            # Align the GCS destination extension with the actual container.
            gcs_ext = os.path.splitext(output_gcs_path)[1].lower()
            if gcs_ext != input_ext:
                output_gcs_path = output_gcs_path[: -len(gcs_ext)] + input_ext

            download_file(video_bucket, video_path, local_video)
            download_file(self.bucket_name, srt_gcs_path, local_srt)

            mov_language = _ISO_639_1_TO_2.get(language_code, language_code)
            cmd = [
                settings.ffmpeg_bin, "-y",
                "-i", local_video,
                "-i", local_srt,
                "-c:v", "copy",
                "-c:a", "copy",
                "-c:s", "mov_text",
                "-metadata:s:s:0", f"language={mov_language}",
                "-disposition:s:0", "default",
                local_output,
            ]

            logger.info(f"Muxing soft subtitles: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg mux failed: {result.stderr[-500:]}")

            return upload_file(self.bucket_name, output_gcs_path, local_output)
