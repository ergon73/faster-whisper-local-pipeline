from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Optional, Sequence, Callable

try:
    from faster_whisper import WhisperModel
except ImportError as exc:  # pragma: no cover - пакет может быть недоступен в окружении
    logging.getLogger(__name__).warning(
        "Не удалось импортировать faster_whisper: %s", exc
    )
    WhisperModel = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


DEFAULT_VIDEO_FORMATS: tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
)

DEFAULT_AUDIO_FORMATS: tuple[str, ...] = (
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
)


TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_BIT_DEPTH = 16
TARGET_PCM_BIT_RATE = TARGET_SAMPLE_RATE * TARGET_CHANNELS * TARGET_BIT_DEPTH


@dataclass(slots=True)
class PipelineConfig:
    """Настройки пайплайна обработки мультимедиа."""

    video_in: Path = Path("video_in")
    audio_in: Path = Path("audio_in")
    audio_out: Path = Path("audio_out")
    transcripts_out: Path = Path("transcribe")
    video_formats: tuple[str, ...] = field(default_factory=lambda: DEFAULT_VIDEO_FORMATS)
    audio_formats: tuple[str, ...] = field(default_factory=lambda: DEFAULT_AUDIO_FORMATS)
    whisper_model: str = "small"
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    cpu_threads: int = 0
    num_workers: int = 1
    skip_fresh_audio: bool = True

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Создает конфигурацию из переменных окружения (.env поддерживается)."""
        if load_dotenv is not None:
            load_dotenv()

        def _parse_extensions(
            value: Optional[str], default: Sequence[str]
        ) -> tuple[str, ...]:
            if not value:
                return tuple(default)
            parsed = tuple(
                f".{ext.lower().lstrip('.')}" for ext in value.split(",") if ext.strip()
            )
            return parsed or tuple(default)

        def _parse_bool(value: Optional[str], default: bool) -> bool:
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        def _parse_int(value: Optional[str], default: int) -> int:
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default

        return cls(
            video_in=Path(os.getenv("VIDEO_IN", "video_in")),
            audio_in=Path(os.getenv("AUDIO_IN", "audio_in")),
            audio_out=Path(os.getenv("AUDIO_OUT", "audio_out")),
            transcripts_out=Path(os.getenv("TRANSCRIBE_OUT", "transcribe")),
            video_formats=_parse_extensions(os.getenv("VIDEO_FORMATS"), DEFAULT_VIDEO_FORMATS),
            audio_formats=_parse_extensions(os.getenv("AUDIO_FORMATS"), DEFAULT_AUDIO_FORMATS),
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            beam_size=_parse_int(os.getenv("WHISPER_BEAM_SIZE"), 5),
            vad_filter=_parse_bool(os.getenv("WHISPER_VAD_FILTER"), True),
            vad_min_silence_ms=_parse_int(os.getenv("WHISPER_VAD_MIN_SILENCE"), 500),
            cpu_threads=max(0, _parse_int(os.getenv("WHISPER_CPU_THREADS"), 0)),
            num_workers=max(1, _parse_int(os.getenv("WHISPER_NUM_WORKERS"), 1)),
            skip_fresh_audio=_parse_bool(os.getenv("SKIP_FRESH_AUDIO"), True),
        )


@dataclass(slots=True)
class TranscriptionJob:
    """Описывает задачу на транскрибацию конкретного файла."""

    source_path: Path
    audio_path: Path


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Создает логгер по умолчанию для CLI-скрипта."""
    logger = logging.getLogger("faster_whisper_local")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_directories(config: PipelineConfig) -> None:
    """Создает выходные директории при необходимости."""
    for folder in (config.audio_out, config.transcripts_out):
        folder.mkdir(parents=True, exist_ok=True)


def ensure_ffmpeg_available(logger: logging.Logger) -> None:
    """Проверяет наличие ffmpeg в системе."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:  # ffmpeg отсутствует
        logger.error("FFmpeg не найден. Установите FFmpeg и добавьте его в PATH.")
        raise RuntimeError("ffmpeg executable not found") from exc
    except subprocess.CalledProcessError as exc:
        logger.error("FFmpeg найден, но возвращает ошибку: %s", exc)
        raise RuntimeError("ffmpeg is not operational") from exc


def _is_audio_up_to_date(source: Path, destination: Path) -> bool:
    """Возвращает True, если файл назначения новее источника."""
    return destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fmt_srt_time(timestamp: float) -> str:
    """Преобразует время в формате float секунд в строку SRT."""
    milliseconds = int(round(timestamp * 1000))
    delta = timedelta(milliseconds=milliseconds)
    base = str(delta)
    if "." in base:
        base_part, frac = base.split(".")
        return f"{base_part.zfill(8)},{frac[:3].ljust(3, '0')}"
    return f"{base.zfill(8)},000"


def _write_srt(segments: list[dict[str, object]], path: Path) -> None:
    """Записывает сегменты в формате SRT."""
    with path.open("w", encoding="utf-8") as file:
        for index, segment in enumerate(segments, start=1):
            file.write(f"{index}\n")
            file.write(
                f"{_fmt_srt_time(float(segment['start']))} --> "
                f"{_fmt_srt_time(float(segment['end']))}\n"
            )
            file.write(str(segment["text"]).strip() + "\n\n")


def _write_jsonl(
    segments: list[dict[str, object]], info: dict[str, object], path: Path
) -> None:
    """Записывает метаданные и сегменты в формате JSONL."""
    with path.open("w", encoding="utf-8") as file:
        header = {"type": "metadata", **info}
        file.write(json.dumps(header, ensure_ascii=False) + "\n")
        for segment in segments:
            file.write(
                json.dumps({"type": "segment", **segment}, ensure_ascii=False) + "\n"
            )


def _probe_audio_stream(source: Path, logger: logging.Logger) -> dict[str, int | str | None] | None:
    """Возвращает метаданные первого аудиопотока средствами ffprobe."""
    arguments = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,channels,sample_rate,bit_rate",
        "-of",
        "json",
        str(source),
    ]

    try:
        result = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        logger.debug("ffprobe не найден — пропускаем сбор метаданных для %s", source.name)
        return None
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        logger.debug("ffprobe завершился с ошибкой для %s: %s", source.name, stderr)
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.debug("Не удалось разобрать вывод ffprobe для %s: %s", source.name, exc)
        return None

    streams = payload.get("streams")
    if not streams:
        return None

    stream = streams[0]
    return {
        "codec_name": stream.get("codec_name"),
        "channels": _safe_int(stream.get("channels")),
        "sample_rate": _safe_int(stream.get("sample_rate")),
        "bit_rate": _safe_int(stream.get("bit_rate")),
    }


def media_requires_ffmpeg(config: PipelineConfig) -> bool:
    """Возвращает True, если для текущих входных данных потребуется FFmpeg.

    Проверяются видео и несжатые аудио форматы. Если включён skip_fresh_audio и
    актуальный WAV уже существует, FFmpeg не требуется для конкретного файла.
    """
    # Видео
    if config.video_in.exists() and config.video_in.is_dir():
        for file in sorted(config.video_in.iterdir()):
            if not file.is_file() or file.suffix.lower() not in config.video_formats:
                continue
            audio_path = config.audio_out / _build_audio_filename(file, "video")
            if config.skip_fresh_audio and _is_audio_up_to_date(file, audio_path):
                continue
            return True

    # Аудио, требующее перекодирования (не WAV)
    if config.audio_in.exists() and config.audio_in.is_dir():
        for file in sorted(config.audio_in.iterdir()):
            if not file.is_file() or file.suffix.lower() not in config.audio_formats:
                continue
            if file.suffix.lower() == ".wav":
                continue
            audio_path = config.audio_out / _build_audio_filename(file, "audio")
            if config.skip_fresh_audio and _is_audio_up_to_date(file, audio_path):
                continue
            return True

    return False


def _build_audio_filename(source: Path, origin_tag: str) -> str:
    """Формирует имя аудиофайла с учетом источника."""
    safe_tag = origin_tag.strip().replace(" ", "_").lower()
    suffix = source.suffix.lower().lstrip(".") or "raw"
    return f"{source.stem}_{safe_tag}_{suffix}.wav"


def extract_audio(
    source: Path,
    destination: Path,
    logger: logging.Logger,
) -> bool:
    """Извлекает или перекодирует аудио трек в формат WAV."""
    metadata = _probe_audio_stream(source, logger)
    if metadata:
        codec = metadata.get("codec_name") or "неизвестно"
        channels = metadata.get("channels")
        sample_rate = metadata.get("sample_rate")
        bit_rate = metadata.get("bit_rate")
        source_kbps = f"{bit_rate / 1000:.0f}" if bit_rate else "?"
        logger.info(
            "Исходное аудио: codec=%s, channels=%s, sample_rate=%s Гц, bitrate=%s kbps",
            codec,
            channels if channels is not None else "?",
            sample_rate if sample_rate is not None else "?",
            source_kbps,
        )

        if bit_rate and bit_rate > 0:
            target_kbps = TARGET_PCM_BIT_RATE / 1000
            if bit_rate <= TARGET_PCM_BIT_RATE:
                logger.info(
                    "Оценка потери качества при конвертации: ~0%% (битрейт %.0f → %.0f kbps).",
                    bit_rate / 1000,
                    target_kbps,
                )
            else:
                loss_pct = (1 - TARGET_PCM_BIT_RATE / bit_rate) * 100
                logger.info(
                    "Оценка потери качества при конвертации: ~%.1f%% (битрейт %.0f → %.0f kbps).",
                    loss_pct,
                    bit_rate / 1000,
                    target_kbps,
                )
        else:
            logger.info(
                "Оценка потери качества при конвертации: недостаточно данных о битрейте исходного аудио."
            )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        "-y",
        str(destination),
    ]

    logger.debug("Запуск ffmpeg: %s", " ".join(command))
    start_time = time.perf_counter()

    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
        ) as proc:
            assert proc.stdout is not None  # для типизации
            for line in proc.stdout:
                logger.info("[ffmpeg] %s", line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    except FileNotFoundError as exc:
        logger.error("FFmpeg не найден при попытке обработки %s", source.name)
        return False
    except RuntimeError as exc:
        logger.error("FFmpeg завершился с ошибкой: %s", exc)
        return False
    except subprocess.SubprocessError as exc:
        logger.error("Не удалось запустить ffmpeg: %s", exc)
        return False

    duration = time.perf_counter() - start_time
    logger.info(
        "Аудио подготовлено: %s -> %s за %.2f сек",
        source.name,
        destination.name,
        duration,
    )
    return True


def copy_wav(source: Path, destination: Path, logger: logging.Logger) -> bool:
    """Копирует WAV-файл в целевую директорию с логированием."""
    try:
        shutil.copy2(source, destination)
    except (OSError, shutil.Error) as exc:
        logger.error("Ошибка копирования %s: %s", source.name, exc)
        return False

    logger.info("Аудио скопировано: %s -> %s", source.name, destination.name)
    return True


def _audio_prepare_handler(source: Path, destination: Path, logger: logging.Logger) -> bool:
    """Выбирает стратегию подготовки аудио: копирование WAV или извлечение через FFmpeg."""
    if source.suffix.lower() == ".wav":
        return copy_wav(source, destination, logger)
    return extract_audio(source, destination, logger)


def _collect_jobs(
    source_dir: Path,
    formats: tuple[str, ...],
    origin_tag: str,
    handler: Callable[[Path, Path, logging.Logger], bool],
    audio_out: Path,
    skip_fresh: bool,
    logger: logging.Logger,
) -> list[TranscriptionJob]:
    jobs: list[TranscriptionJob] = []
    for file in sorted(source_dir.iterdir()):
        if not file.is_file() or file.suffix.lower() not in formats:
            continue
        audio_path = audio_out / _build_audio_filename(file, origin_tag)
        if skip_fresh and _is_audio_up_to_date(file, audio_path):
            logger.info(
                "Пропуск подготовки: %s (актуальный аудиофайл уже существует)", file.name
            )
            jobs.append(TranscriptionJob(source_path=file, audio_path=audio_path))
            continue
        if handler(file, audio_path, logger):
            jobs.append(TranscriptionJob(source_path=file, audio_path=audio_path))
    return jobs


def prepare_jobs(config: PipelineConfig, logger: logging.Logger) -> list[TranscriptionJob]:
    """Собирает список задач на транскрибацию."""
    jobs: list[TranscriptionJob] = []

    if config.video_in.exists() and config.video_in.is_dir():
        jobs.extend(
            _collect_jobs(
                config.video_in,
                config.video_formats,
                "video",
                extract_audio,
                config.audio_out,
                config.skip_fresh_audio,
                logger,
            )
        )

    if config.audio_in.exists() and config.audio_in.is_dir():
        jobs.extend(
            _collect_jobs(
                config.audio_in,
                config.audio_formats,
                "audio",
                _audio_prepare_handler,
                config.audio_out,
                config.skip_fresh_audio,
                logger,
            )
        )

    return jobs


def detect_device_preference(logger: logging.Logger) -> tuple[str, str]:
    """Определяет устройство и compute_type для WhisperModel."""
    preferred_device = os.getenv("WHISPER_DEVICE", "").strip().lower()

    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    if preferred_device == "cuda":
        # При явном запросе CUDA пробуем загрузить модель на GPU.
        # При ошибке загрузки произойдёт fallback в load_model().
        return "cuda", "float16"

    elif preferred_device == "cpu":
        return "cpu", "int8"

    if _cuda_available():
        return "cuda", "float16"

    return "cpu", os.getenv("WHISPER_COMPUTE_TYPE", "int8")


def load_model(
    config: PipelineConfig, logger: logging.Logger
) -> WhisperModel:
    """Загружает модель Whisper с автоматическим переключением устройства."""
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper не установлен для текущего окружения. Установите пакет или используйте Python 3.11/3.12, где доступны колёса зависимостей."
        )
    logger.info("Загрузка модели Whisper '%s'...", config.whisper_model)
    device, compute_type = detect_device_preference(logger)

    try:
        model = WhisperModel(
            config.whisper_model,
            device=device,
            compute_type=compute_type,
            cpu_threads=config.cpu_threads,
            num_workers=config.num_workers,
        )
        logger.info("Модель '%s' загружена на %s (compute_type=%s).", config.whisper_model, device, compute_type)
        return model
    except Exception as error:
        if device == "cuda":
            logger.warning(
                "Не удалось загрузить модель на CUDA (%s). Пробуем CPU (int8).",
                error,
            )
            return WhisperModel(
                config.whisper_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=config.cpu_threads,
                num_workers=config.num_workers,
            )
        raise


def transcribe_audio(
    job: TranscriptionJob,
    output_path: Path,
    model: WhisperModel,
    config: PipelineConfig,
    logger: logging.Logger,
) -> bool:
    """Выполняет транскрибацию аудиофайла и сохраняет результат."""
    logger.info("Начало транскрибации: %s", job.audio_path.name)
    start_time = time.perf_counter()

    try:
        segments_iter, info = model.transcribe(
            str(job.audio_path),
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
            vad_parameters={"min_silence_duration_ms": config.vad_min_silence_ms},
        )

        segments_data: list[dict[str, object]] = []
        with output_path.open("w", encoding="utf-8") as target_txt:
            for segment in segments_iter:
                text = segment.text.strip()
                if not text:
                    continue
                target_txt.write(text + "\n")
                segments_data.append(
                    {"start": float(segment.start), "end": float(segment.end), "text": text}
                )

        metadata = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "source_audio": job.audio_path.name,
        }

        _write_srt(segments_data, output_path.with_suffix(".srt"))
        _write_jsonl(segments_data, metadata, output_path.with_suffix(".jsonl"))

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Транскрибация завершена: %s (%.2f сек, язык=%s, p=%.2f, длительность=%.2f сек)",
            job.audio_path.name,
            elapsed,
            info.language,
            info.language_probability,
            info.duration,
        )
        return True
    except Exception as exc:
        logger.exception("Ошибка транскрибации %s", job.audio_path.name)
        return False


def process_files(config: PipelineConfig, logger: logging.Logger) -> None:
    """Основной сценарий обработки пользовательских файлов."""
    logger.info("Запуск пайплайна Faster-Whisper Local")
    ensure_directories(config)
    if media_requires_ffmpeg(config):
        try:
            ensure_ffmpeg_available(logger)
        except RuntimeError as exc:
            logger.error("Нужен FFmpeg для видео/сжатых аудио: %s", exc)
            return

    jobs = prepare_jobs(config, logger)
    if not jobs:
        logger.warning(
            "Не найдено файлов для обработки. Поместите данные в %s или %s.",
            config.video_in,
            config.audio_in,
        )
        return

    logger.info("Найдено задач для транскрибации: %d", len(jobs))

    try:
        model = load_model(config, logger)
    except Exception as exc:
        logger.error("Не удалось загрузить модель Whisper: %s", exc)
        return

    success = 0
    for index, job in enumerate(jobs, start=1):
        logger.info("[%d/%d] Обработка %s", index, len(jobs), job.audio_path.name)
        result_path = config.transcripts_out / f"{job.audio_path.stem}.txt"
        if result_path.exists() and result_path.stat().st_mtime >= job.audio_path.stat().st_mtime:
            logger.info("Пропуск транскрибации (актуальный TXT уже есть): %s", result_path.name)
            success += 1
            continue
        if transcribe_audio(job, result_path, model, config, logger):
            success += 1
            logger.info("Результат сохранен: %s", result_path.name)
        else:
            logger.error("Ошибка при обработке %s", job.audio_path.name)

    logger.info(
        "Обработка завершена: %d/%d файлов успешно.",
        success,
        len(jobs),
    )
    logger.info("Транскрипты: %s | Аудио: %s", config.transcripts_out, config.audio_out)


def main() -> None:
    """CLI-точка входа."""
    logger = configure_logging()
    config = PipelineConfig.from_env()
    try:
        process_files(config, logger)
    except Exception as exc:  # pragma: no cover - защита от непредвиденных ошибок
        logger.exception("Необработанная ошибка исполнения: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
