"""Production-тест финальной конфигурации v2 (с улучшенным промптом) на всех 4 днях.

VERSION: Production Final v2 (Enhanced Prompt)
DATE: 2025-11-26

КОНФИГУРАЦИЯ:
- Модель: qwen3:32b
- Промпт: Balanced + Fallback для плотного технического контента
- Параметры: temp=0.3, repeat_penalty=1.1, top_k=40
- Пост-обработка: regex-очистка (<think>, markdown-обертка)

УЛУЧШЕНИЯ v2:
- Добавлена fallback-инструкция для обработки плотного технического контента
- Решает проблему преждевременной остановки генерации (главы 8-12 День 3)

ЦЕЛЬ: Обработать все 65 глав из 4 дней
"""

import json
import time
import re
from pathlib import Path
from datetime import datetime

try:
    import ollama
except ImportError:
    print("ERROR: pip install ollama")
    exit(1)

# === ФИНАЛЬНАЯ КОНФИГУРАЦИЯ v2 ===

MODEL_NAME = "qwen3:32b"

SYSTEM_PROMPT = "You are a professional technical blog editor. Output clean Markdown only."

MODEL_OPTIONS = {
    "temperature": 0.3,
    "repeat_penalty": 1.1,
    "top_k": 40,
    "num_ctx": 8192,
    "num_predict": 3000
}

# === Balanced Prompt v2 (с fallback-инструкцией) ===

def build_prompt_balanced_v2(chapter_title: str, chapter_text: str) -> str:
    """Промпт для сбалансированной модели (редактор Хабра) + fallback для плотной техники."""
    return f"""Ты — редактор технического блога на Хабре.
Твоя задача: превратить расшифровку доклада в увлекательную, читаемую статью.

**ТВОЯ ЦЕЛЬ:**
Сделать текст понятным для Junior/Middle разработчиков, сохранив глубину материала.

**ИНСТРУКЦИИ:**
1. **ТЕРМИНОЛОГИЯ:**
   - Исправляй ошибки STT: "пайторч" -> `PyTorch`, "джипити" -> `GPT`, "пандас" -> `Pandas`.
   - Всегда используй правильный регистр для библиотек (не numpy, а `NumPy`).
   - Код и названия библиотек оборачивай в обратные кавычки (`code`).

2. **СТИЛЬ И СТРУКТУРА:**
   - Разбей текст на логические секции с заголовками (##, ###).
   - Если спикер приводит пример или метафору — **сохрани её**, это делает текст живым.
   - Избегай канцеляризмов. Пиши просто и емко.
   - Убирай только явный мусор ("э-э-э", "слышно меня"), но не суши текст до состояния справочника.

3. **ФОРМАТ:**
   - Только Markdown.
   - Используй списки для перечислений.
   - Важные мысли выделяй **жирным**.

**ОСОБЫЙ СЛУЧАЙ (техническая лекция без контекста):**
Если исходный текст сразу начинается с плотной технической терминологии без вступления:
- Создай краткое вступление (1-2 абзаца), объясняющее контекст
- Структурируй материал в виде пошагового руководства или списка ключевых концепций
- Добавь промежуточные пояснения между техническими блоками
- Даже если исходный текст сухой, постарайся сделать его максимально читаемым

**ВХОДНОЙ ТЕКСТ:**
Тема: {chapter_title}
Текст:
{chapter_text}

**СТАТЬЯ (MARKDOWN):**"""


def clean_output(text: str) -> str:
    """
    Пост-обработка с regex-очисткой (страховка).

    1. Удаляет <think>...</think> (на случай философствования)
    2. Извлекает текст из Markdown-обертки (```markdown...```)
    """
    # 1. Удаляем <think> теги
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. Извлекаем из Markdown-обертки (Qwen любит оборачивать!)
    match = re.search(r'```(?:markdown)?\s*(.*?)```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    return text.strip()


# === Файлы для обработки ===

DAYS = [
    {
        "name": "День 1",
        "chapters_file": "transcribe/Большой марафон по Классическому AI. День 1._video_mp4_chapters.json",
        "packed_file": "transcribe/Большой марафон по Классическому AI. День 1._video_mp4_packed.jsonl"
    },
    {
        "name": "День 2",
        "chapters_file": "transcribe/Большой марафон по Классическому AI. День 2._video_mp4_chapters.json",
        "packed_file": "transcribe/Большой марафон по Классическому AI. День 2._video_mp4_packed.jsonl"
    },
    {
        "name": "День 3",
        "chapters_file": "transcribe/Большой марафон по Классическому AI. День 3_video_mp4_chapters.json",
        "packed_file": "transcribe/Большой марафон по Классическому AI. День 3_video_mp4_packed.jsonl"
    },
    {
        "name": "День 4",
        "chapters_file": "transcribe/Большой марафон по Классическому AI. День 4._video_mp4_chapters.json",
        "packed_file": "transcribe/Большой марафон по Классическому AI. День 4._video_mp4_packed.jsonl"
    }
]

# === Функции ===

def load_day_data(day_info: dict) -> tuple[list, dict]:
    """Загружает все главы и абзацы для одного дня."""
    chapters_file = Path(day_info["chapters_file"])
    packed_file = Path(day_info["packed_file"])

    with chapters_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    chapters = data["chapters"]

    # Загружаем все абзацы
    paragraphs = {}
    with packed_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("type") == "paragraph":
                paragraphs[row["id"]] = row["text"]

    return chapters, paragraphs


def process_chapter(chapter: dict, paragraphs: dict, day_name: str, chapter_idx: int) -> dict:
    """Обрабатывает одну главу."""
    chapter_title = chapter["title"]

    # Собираем текст главы
    chapter_text = "\n\n".join(
        paragraphs[pid] for pid in chapter["paragraph_ids"] if pid in paragraphs
    )

    print(f"\n{'='*80}")
    print(f"{day_name}, Глава {chapter_idx + 1}: {chapter_title}")
    print(f"Исходная длина: {len(chapter_text)} символов")

    prompt = build_prompt_balanced_v2(chapter_title, chapter_text)
    start_time = time.time()

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options=MODEL_OPTIONS,
            think=False
        )

        edited_text = response['message']['content'].strip()

        # Применяем пост-обработку (страховка)
        edited_text = clean_output(edited_text)

        elapsed_time = time.time() - start_time
        compression_ratio = len(edited_text) / len(chapter_text)

        print(f"[OK] Время: {elapsed_time:.2f} сек")
        print(f"  Новая длина: {len(edited_text)} символов")
        print(f"  Компрессия: {compression_ratio:.2%} (удалено {(1-compression_ratio)*100:.1f}%)")

        return {
            "day": day_name,
            "chapter_idx": chapter_idx,
            "title": chapter_title,
            "success": True,
            "time": round(elapsed_time, 2),
            "original_length": len(chapter_text),
            "edited_length": len(edited_text),
            "compression_ratio": round(compression_ratio, 4),
            "edited_text": edited_text,
            "error": None
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[ERROR] {e}")

        return {
            "day": day_name,
            "chapter_idx": chapter_idx,
            "title": chapter_title,
            "success": False,
            "time": round(elapsed_time, 2),
            "original_length": len(chapter_text),
            "edited_length": 0,
            "compression_ratio": 0,
            "edited_text": "",
            "error": str(e)
        }


def main():
    """Основная функция."""
    print("=== PRODUCTION-ТЕСТ v2: Qwen3:32b + Balanced + Enhanced Fallback ===")
    print(f"Модель: {MODEL_NAME}")
    print(f"Дней: {len(DAYS)}")

    # Подсчитываем общее количество глав
    total_chapters = 0
    for day_info in DAYS:
        with open(day_info["chapters_file"], "r", encoding="utf-8") as f:
            data = json.load(f)
            total_chapters += len(data["chapters"])

    print(f"Всего глав: {total_chapters}")
    print(f"Ожидаемое время: ~{total_chapters * 28 / 60:.1f} минут")

    print("\n=== Начинаем обработку ===\n")

    all_results = []
    day_summaries = []

    for day_info in DAYS:
        print(f"\n{'#'*80}")
        print(f"# {day_info['name']}")
        print(f"{'#'*80}")

        chapters, paragraphs = load_day_data(day_info)

        day_results = []
        for idx, chapter in enumerate(chapters):
            result = process_chapter(chapter, paragraphs, day_info["name"], idx)
            day_results.append(result)
            all_results.append(result)

            # Сохраняем отредактированный текст
            if result["success"]:
                output_file = Path(f"production_output_v2/{day_info['name']}_chapter_{idx+1}.md")
                output_file.parent.mkdir(exist_ok=True)
                output_file.write_text(result["edited_text"], encoding="utf-8")

        # Статистика по дню
        successful = [r for r in day_results if r["success"]]
        if successful:
            avg_time = sum(r["time"] for r in successful) / len(successful)
            avg_compression = sum(r["compression_ratio"] for r in successful) / len(successful)

            day_summary = {
                "day": day_info["name"],
                "total": len(day_results),
                "successful": len(successful),
                "avg_time": round(avg_time, 2),
                "avg_compression": round(avg_compression, 4)
            }
            day_summaries.append(day_summary)

            print(f"\n[{day_info['name']}] Успешно: {len(successful)}/{len(day_results)}")
            print(f"  Средняя скорость: {avg_time:.2f} сек/глава")
            print(f"  Средняя компрессия: {avg_compression:.2%}")

    # Сохраняем полный отчёт
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(f"production_output_v2/production_report_v2_{timestamp}.json")
    report_file.parent.mkdir(exist_ok=True)

    with report_file.open("w", encoding="utf-8") as f:
        json.dump({
            "version": "production-final-v2-enhanced",
            "timestamp": timestamp,
            "model": MODEL_NAME,
            "config": {
                "prompt": "Balanced v2 (Habr editor + Enhanced Fallback)",
                "options": MODEL_OPTIONS
            },
            "total_chapters": total_chapters,
            "day_summaries": day_summaries,
            "all_results": all_results
        }, f, indent=2, ensure_ascii=False)

    # Итоговая статистика
    print(f"\n{'='*80}")
    print("[PRODUCTION v2] ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*80}\n")

    successful = [r for r in all_results if r["success"]]
    print(f"Обработано глав: {len(successful)}/{total_chapters}")

    if successful:
        avg_time = sum(r["time"] for r in successful) / len(successful)
        avg_compression = sum(r["compression_ratio"] for r in successful) / len(successful)
        total_time = sum(r["time"] for r in successful)

        print(f"Средняя скорость: {avg_time:.2f} сек/глава")
        print(f"Средняя компрессия: {avg_compression:.2%} (удалено {(1-avg_compression)*100:.1f}%)")
        print(f"Общее время: {total_time / 60:.1f} минут")

        # Анализ аномалий
        anomalies_low = [r for r in successful if r["compression_ratio"] < 0.15]
        anomalies_high = [r for r in successful if r["compression_ratio"] > 1.50]

        print(f"\nАНОМАЛИИ:")
        print(f"  Низкая компрессия (<15%): {len(anomalies_low)} глав")
        print(f"  Расширение (>150%): {len(anomalies_high)} глав")

        if anomalies_low:
            print(f"\n  Главы с низкой компрессией:")
            for r in anomalies_low:
                print(f"    - {r['day']}, Глава {r['chapter_idx']+1}: {r['compression_ratio']:.2%}")

    print(f"\n[REPORT] Отчёт сохранён: {report_file}")
    print(f"[FILES] Отредактированные главы: production_output_v2/")
    print("\n=== Production-тест v2 завершён! ===")


if __name__ == "__main__":
    main()
