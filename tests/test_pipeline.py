import logging
import os
from pathlib import Path

import pytest

import transcribe_v2 as pipeline


@pytest.fixture
def test_logger() -> logging.Logger:
    """Возвращает изолированный логгер для тестов."""
    logger = logging.getLogger("test_pipeline")
    logger.handlers.clear()
    handler = logging.NullHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def test_build_audio_filename_adds_origin_tag() -> None:
    source = Path("video_in") / "meeting.mp4"
    assert pipeline._build_audio_filename(source, "video") == "meeting_video.wav"
    assert pipeline._build_audio_filename(source, "AUDIO") == "meeting_audio.wav"


def test_prepare_jobs_creates_unique_audio_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_logger: logging.Logger) -> None:
    video_dir = tmp_path / "video_in"
    audio_dir = tmp_path / "audio_in"
    audio_out = tmp_path / "audio_out"
    transcripts = tmp_path / "transcribe"

    for directory in (video_dir, audio_dir, audio_out, transcripts):
        directory.mkdir(parents=True, exist_ok=True)

    video_file = video_dir / "meeting.mp4"
    video_file.write_bytes(b"video")
    audio_file = audio_dir / "meeting.wav"
    audio_file.write_bytes(b"audio")

    calls: list[tuple[str, str, str]] = []

    def fake_extract(source: Path, destination: Path, logger: logging.Logger) -> bool:
        destination.touch()
        calls.append(("extract", source.name, destination.name))
        return True

    def fake_copy(source: Path, destination: Path, logger: logging.Logger) -> bool:
        destination.touch()
        calls.append(("copy", source.name, destination.name))
        return True

    monkeypatch.setattr(pipeline, "extract_audio", fake_extract)
    monkeypatch.setattr(pipeline, "copy_wav", fake_copy)

    config = pipeline.PipelineConfig(
        video_in=video_dir,
        audio_in=audio_dir,
        audio_out=audio_out,
        transcripts_out=transcripts,
        skip_fresh_audio=False,
        whisper_model="tiny",
    )

    jobs = pipeline.prepare_jobs(config, test_logger)

    filenames = {job.audio_path.name for job in jobs}

    assert filenames == {"meeting_video.wav", "meeting_audio.wav"}
    assert ("extract", "meeting.mp4", "meeting_video.wav") in calls
    assert ("copy", "meeting.wav", "meeting_audio.wav") in calls


def test_prepare_jobs_skips_fresh_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, test_logger: logging.Logger) -> None:
    audio_dir = tmp_path / "audio_in"
    audio_out = tmp_path / "audio_out"
    transcripts = tmp_path / "transcribe"

    for directory in (audio_dir, audio_out, transcripts):
        directory.mkdir(parents=True, exist_ok=True)

    source_audio = audio_dir / "sample.mp3"
    source_audio.write_bytes(b"mp3")

    destination_audio = audio_out / "sample_audio.wav"
    destination_audio.write_bytes(b"wav")

    src_stat = source_audio.stat()
    os.utime(destination_audio, (src_stat.st_atime + 10, src_stat.st_mtime + 10))

    called = {"extract": False}

    def fail_extract(*args, **kwargs) -> bool:
        called["extract"] = True
        return False

    monkeypatch.setattr(pipeline, "extract_audio", fail_extract)
    monkeypatch.setattr(pipeline, "copy_wav", fail_extract)

    config = pipeline.PipelineConfig(
        video_in=tmp_path / "video_in",
        audio_in=audio_dir,
        audio_out=audio_out,
        transcripts_out=transcripts,
        skip_fresh_audio=True,
        whisper_model="tiny",
    )

    jobs = pipeline.prepare_jobs(config, test_logger)

    assert len(jobs) == 1, "Файл должен быть добавлен в очередь без повторной конвертации"
    assert jobs[0].audio_path.name == "sample_audio.wav"
    assert called["extract"] is False, "Повторная конвертация не должна выполняться для свежего WAV"

