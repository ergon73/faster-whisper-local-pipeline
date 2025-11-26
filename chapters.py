"""Этап 2: Разбивка транскриптов на главы с автоматическими названиями."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

try:
    import ollama
except ImportError:
    print("ERROR: Установите ollama: pip install ollama")
    sys.exit(1)

DEFAULT_TRANSCRIBE_OUT = "transcribe"

def getenv_str(key: str, default: str) -> str:
    """Возвращает строку из переменной окружения или значение по умолчанию."""
    v = os.getenv(key)
    return v if v is not None and v.strip() else default

def getenv_int(key: str, default: int) -> int:
    """Возвращает целое число из окружения с запасным значением."""
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def getenv_float(key: str, default: float) -> float:
    """Возвращает float из окружения с запасным значением."""
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

def getenv_bool(key: str, default: bool) -> bool:
    """Интерпретирует булевые переменные окружения."""
    v = os.getenv(key)
    if v is None: return default
    return v.strip().lower() in {"1","true","yes","y","on"}

@dataclass
class Paragraph:
    """Абзац из packed.jsonl."""
    id: int
    start: Optional[float]
    end: Optional[float]
    text: str

@dataclass
class Chapter:
    """Глава - группа абзацев с названием."""
    id: int
    title: str
    start: Optional[float]
    end: Optional[float]
    paragraphs: List[Paragraph]

    @property
    def duration(self) -> Optional[float]:
        """Длительность главы в секундах."""
        if self.start is not None and self.end is not None:
            return self.end - self.start
        return None

    @property
    def text(self) -> str:
        """Полный текст главы."""
        return "\n\n".join(p.text for p in self.paragraphs)

def read_packed_jsonl(path: Path) -> tuple[dict, list[Paragraph]]:
    """Читает packed.jsonl файл и возвращает метаданные и абзацы."""
    meta = {}
    paragraphs: List[Paragraph] = []

    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line: continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            t = row.get("type")
            if first and t == "metadata":
                meta = row
                first = False
                continue

            if t == "paragraph":
                paragraphs.append(Paragraph(
                    id=row.get("id", 0),
                    start=row.get("start"),
                    end=row.get("end"),
                    text=row.get("text", "")
                ))

            first = False

    return meta, paragraphs

def split_into_chapters(
    paragraphs: List[Paragraph],
    min_gap_sec: float,
    min_duration_sec: float,
    max_duration_sec: float,
    min_paragraphs: int
) -> List[List[Paragraph]]:
    """
    Разбивает абзацы на главы по эвристикам.

    Правила:
    1. Разделяем при паузе >= min_gap_sec
    2. Главы не короче min_duration_sec
    3. Главы не длиннее max_duration_sec (мягкое ограничение)
    4. Главы содержат >= min_paragraphs абзацев
    """
    if not paragraphs:
        return []

    chapters: List[List[Paragraph]] = []
    current_chapter: List[Paragraph] = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        prev = paragraphs[i-1]
        curr = paragraphs[i]

        # Вычисляем паузу
        gap = None
        if prev.end is not None and curr.start is not None:
            gap = curr.start - prev.end

        # Вычисляем длительность текущей главы
        chapter_duration = None
        if current_chapter[0].start is not None and prev.end is not None:
            chapter_duration = prev.end - current_chapter[0].start

        # Условия разделения:
        # 1. Большая пауза
        big_gap = gap is not None and gap >= min_gap_sec

        # 2. Глава слишком длинная
        too_long = chapter_duration is not None and chapter_duration >= max_duration_sec

        # Разделяем если есть большая пауза ИЛИ глава слишком длинная
        should_split = big_gap or too_long

        # НО не разделяем если текущая глава слишком короткая
        if should_split and chapter_duration is not None:
            if chapter_duration < min_duration_sec or len(current_chapter) < min_paragraphs:
                should_split = False

        if should_split:
            # Сохраняем текущую главу
            chapters.append(current_chapter)
            # Начинаем новую
            current_chapter = [curr]
        else:
            # Добавляем в текущую главу
            current_chapter.append(curr)

    # Добавляем последнюю главу
    if current_chapter:
        chapters.append(current_chapter)

    # Слияние слишком маленьких глав
    merged_chapters: List[List[Paragraph]] = []
    for chapter_paras in chapters:
        # Вычисляем длительность
        duration = None
        if chapter_paras[0].start is not None and chapter_paras[-1].end is not None:
            duration = chapter_paras[-1].end - chapter_paras[0].start

        # Если глава слишком короткая и есть предыдущая - сливаем
        if merged_chapters and (
            (duration is not None and duration < min_duration_sec) or
            len(chapter_paras) < min_paragraphs
        ):
            merged_chapters[-1].extend(chapter_paras)
        else:
            merged_chapters.append(chapter_paras)

    return merged_chapters

def generate_chapter_title(chapter_text: str, chapter_num: int, llm_model: str, logger: logging.Logger) -> str:
    """Генерирует название главы через LLM (Qwen3-32B)."""

    # Ограничиваем текст для LLM (первые 2000 символов достаточно)
    sample_text = chapter_text[:2000]
    if len(chapter_text) > 2000:
        sample_text += "..."

    messages = [
        {
            "role": "system",
            "content": "Ты эксперт по анализу образовательного видео. Даёшь короткие точные названия (3-7 слов) для фрагментов лекций."
        },
        # Few-shot примеры
        {
            "role": "user",
            "content": "Дай короткое название (3-7 слов):\n\nСегодня мы разберём основы линейной регрессии. Это базовый алгоритм машинного обучения для предсказания непрерывных значений."
        },
        {
            "role": "assistant",
            "content": "Основы линейной регрессии"
        },
        {
            "role": "user",
            "content": "Дай короткое название (3-7 слов):\n\nВ этой части установим PyTorch и создадим первую нейронную сеть. Разберём архитектуру, функцию потерь и процесс обучения."
        },
        {
            "role": "assistant",
            "content": "Установка PyTorch и первая нейросеть"
        },
        # Реальный запрос
        {
            "role": "user",
            "content": f"Дай короткое название (3-7 слов) для главы {chapter_num}:\n\n{sample_text}"
        }
    ]

    try:
        logger.info(f"Генерация названия для главы {chapter_num} через {llm_model}...")

        response = ollama.chat(
            model=llm_model,
            messages=messages,
            options={"temperature": 0.3, "num_predict": 50},
            think=False  # Отключаем thinking mode
        )

        title = response['message']['content'].strip()

        # Очистка от кавычек и лишних символов
        title = title.strip('"\'«»"')

        # Если пустое - используем дефолт
        if not title:
            title = f"Глава {chapter_num}"
            logger.warning(f"LLM вернул пустое название для главы {chapter_num}, используем дефолт")

        logger.info(f"Глава {chapter_num}: «{title}»")
        return title

    except Exception as e:
        logger.error(f"Ошибка при генерации названия для главы {chapter_num}: {e}")
        return f"Глава {chapter_num}"

def create_chapters(
    paragraph_groups: List[List[Paragraph]],
    llm_model: str,
    use_llm: bool,
    logger: logging.Logger
) -> List[Chapter]:
    """Создаёт объекты Chapter с названиями."""
    chapters: List[Chapter] = []

    for i, paras in enumerate(paragraph_groups, start=1):
        if not paras:
            continue

        # Определяем границы главы
        start = paras[0].start
        end = paras[-1].end

        # Генерируем название
        if use_llm:
            chapter_text = "\n\n".join(p.text for p in paras)
            title = generate_chapter_title(chapter_text, i, llm_model, logger)
        else:
            title = f"Глава {i}"

        chapters.append(Chapter(
            id=i,
            title=title,
            start=start,
            end=end,
            paragraphs=paras
        ))

    return chapters

def write_chapters_json(chapters: List[Chapter], meta: dict, path: Path) -> None:
    """Сохраняет главы в JSON формате."""
    output = {
        "metadata": meta,
        "chapters": [
            {
                "id": ch.id,
                "title": ch.title,
                "start": ch.start,
                "end": ch.end,
                "duration": ch.duration,
                "paragraph_ids": [p.id for p in ch.paragraphs],
                "paragraph_count": len(ch.paragraphs)
            }
            for ch in chapters
        ]
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

def write_structured_md(chapters: List[Chapter], meta: dict, path: Path) -> None:
    """Создаёт structured markdown с оглавлением и текстом по главам."""
    with path.open("w", encoding="utf-8") as f:
        # Заголовок
        f.write(f"# {meta.get('source_file', 'Транскрипт')}\n\n")

        # Метаданные
        if meta.get("duration"):
            duration_min = meta["duration"] / 60
            f.write(f"**Длительность:** {duration_min:.1f} минут\n\n")

        f.write(f"**Глав:** {len(chapters)}\n\n")

        # Оглавление
        f.write("## 📑 Оглавление\n\n")
        for ch in chapters:
            timestamp = ""
            if ch.start is not None:
                minutes = int(ch.start // 60)
                seconds = int(ch.start % 60)
                timestamp = f" `[{minutes:02d}:{seconds:02d}]`"

            f.write(f"{ch.id}. **{ch.title}**{timestamp}\n")

        f.write("\n---\n\n")

        # Текст по главам
        for ch in chapters:
            # Заголовок главы
            f.write(f"## Глава {ch.id}: {ch.title}\n\n")

            # Таймкод
            if ch.start is not None and ch.end is not None:
                start_min = int(ch.start // 60)
                start_sec = int(ch.start % 60)
                end_min = int(ch.end // 60)
                end_sec = int(ch.end % 60)
                f.write(f"**Таймкод:** {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}\n\n")

            # Текст
            f.write(ch.text)
            f.write("\n\n---\n\n")

def write_report(report: dict, path: Path) -> None:
    """Сохраняет отчёт о главировании."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

def main() -> None:
    """CLI-точка входа для разбивки на главы."""
    if load_dotenv is not None: load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("chapters")

    out_dir = Path(getenv_str("TRANSCRIBE_OUT", DEFAULT_TRANSCRIBE_OUT))
    if not out_dir.exists():
        logger.error("Directory not found: %s", out_dir)
        sys.exit(1)

    # Конфигурация
    cfg = {
        "min_gap_sec": getenv_float("CHAPTER_MIN_GAP_SEC", 8.0),
        "min_duration_sec": getenv_float("CHAPTER_MIN_DURATION_SEC", 120.0),  # 2 минуты
        "max_duration_sec": getenv_float("CHAPTER_MAX_DURATION_SEC", 600.0),  # 10 минут
        "min_paragraphs": getenv_int("CHAPTER_MIN_PARAGRAPHS", 3),
        "llm_model": getenv_str("CHAPTER_LLM_MODEL", "qwen3:32b"),
        "use_llm": getenv_bool("CHAPTER_USE_LLM", True),
        "write_report": getenv_bool("CHAPTER_REPORT", True)
    }

    logger.info("Конфигурация: %s", cfg)

    # Найти все *_packed.jsonl файлы
    packed_files = list(out_dir.glob("*_packed.jsonl"))

    if not packed_files:
        logger.warning("No *_packed.jsonl files found in %s", out_dir)
        return

    for packed_file in sorted(packed_files):
        logger.info("=" * 70)
        logger.info("Обработка: %s", packed_file.name)

        # Читаем файл
        meta, paragraphs = read_packed_jsonl(packed_file)

        if not paragraphs:
            logger.warning("Нет абзацев в %s, пропускаем", packed_file.name)
            continue

        logger.info("Загружено абзацев: %d", len(paragraphs))

        # Разбиваем на главы по эвристикам
        paragraph_groups = split_into_chapters(
            paragraphs,
            min_gap_sec=cfg["min_gap_sec"],
            min_duration_sec=cfg["min_duration_sec"],
            max_duration_sec=cfg["max_duration_sec"],
            min_paragraphs=cfg["min_paragraphs"]
        )

        logger.info("Найдено глав (эвристика): %d", len(paragraph_groups))

        # Создаём главы с названиями
        chapters = create_chapters(
            paragraph_groups,
            llm_model=cfg["llm_model"],
            use_llm=cfg["use_llm"],
            logger=logger
        )

        # Формируем имена выходных файлов
        base_name = packed_file.name.replace("_packed.jsonl", "")
        chapters_json_path = out_dir / f"{base_name}_chapters.json"
        structured_md_path = out_dir / f"{base_name}_structured.md"
        report_path = out_dir / f"{base_name}_chapters_report.json"

        # Сохраняем результаты
        meta["source_file"] = packed_file.name
        write_chapters_json(chapters, meta, chapters_json_path)
        write_structured_md(chapters, meta, structured_md_path)

        # Отчёт
        if cfg["write_report"]:
            report = {
                "source": packed_file.name,
                "total_paragraphs": len(paragraphs),
                "total_chapters": len(chapters),
                "avg_chapter_duration": sum(ch.duration or 0 for ch in chapters) / len(chapters) if chapters else 0,
                "config": cfg,
                "chapters_summary": [
                    {
                        "id": ch.id,
                        "title": ch.title,
                        "duration": ch.duration,
                        "paragraphs": len(ch.paragraphs)
                    }
                    for ch in chapters
                ]
            }
            write_report(report, report_path)

        logger.info("✓ Создано: %s", chapters_json_path.name)
        logger.info("✓ Создано: %s", structured_md_path.name)
        if cfg["write_report"]:
            logger.info("✓ Создано: %s", report_path.name)

        logger.info("Обработано глав: %d", len(chapters))

if __name__ == "__main__":
    main()
