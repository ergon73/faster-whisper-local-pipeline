"""Сравнительное тестирование LLM-моделей для редактирования транскриптов.

VERSION: 2.0 - Optimized Prompts
DATE: 2025-11-26

ИЗМЕНЕНИЯ от v1:
1. SYSTEM_PROMPT переписан на английском с явными запретами (NEGATIVE CONSTRAINTS)
2. build_prompt обновлен:
   - Добавлены NEGATIVE CONSTRAINTS (запреты на вступления, <think> теги)
   - Добавлен ONE-SHOT пример
   - Добавлена Markdown структура (##, ###, списки)
3. Параметры генерации:
   - repeat_penalty: 1.15 (предотвращает зацикливание)
   - num_ctx: 8192 (больше контекста для длинных глав)

ИСТОЧНИК: optimal-prompts-recomended.md
"""

import json
import time
from pathlib import Path
from datetime import datetime

try:
    import ollama
except ImportError:
    print("ERROR: pip install ollama")
    exit(1)

# === Конфигурация ===

MODELS = [
    "qwen3:32b",                           # Qwen3 32B (20GB) - baseline
    "qwen3:30b-a3b",                       # Qwen3 30B MoE (19GB) - 30B total, 3B active
    "gpt-oss:20b",                         # OpenAI GPT-OSS 20B MoE (12GB) - 21B total, 3.6B active, reasoning
    "gemma3:27b-it-qat",                   # Gemma 3 27B IT-QAT (18GB)
]

TEMPERATURE = 0.3
MAX_TOKENS = 3000
REPEAT_PENALTY = 1.15  # NEW: Предотвращает зацикливание
NUM_CTX = 8192         # NEW: Больше контекста для длинных глав

# Выберем 3 главы разного размера для теста
TEST_CHAPTERS = [
    {
        "file": "transcribe/Большой марафон по Классическому AI. День 1._video_mp4_chapters.json",
        "packed": "transcribe/Большой марафон по Классическому AI. День 1._video_mp4_packed.jsonl",
        "chapter_idx": 0,  # Глава 1 - длинная, много маркетинга
        "name": "День 1, Глава 1 (длинная, ~7800 символов)"
    },
    {
        "file": "transcribe/Большой марафон по Классическому AI. День 2._video_mp4_chapters.json",
        "packed": "transcribe/Большой марафон по Классическому AI. День 2._video_mp4_packed.jsonl",
        "chapter_idx": 5,  # Глава 6 - средняя
        "name": "День 2, Глава 6 (средняя, ~3500 символов)"
    },
    {
        "file": "transcribe/Большой марафон по Классическому AI. День 3_video_mp4_chapters.json",
        "packed": "transcribe/Большой марафон по Классическому AI. День 3_video_mp4_packed.jsonl",
        "chapter_idx": 2,  # Глава 3 - короткая
        "name": "День 3, Глава 3 (короткая, ~2000 символов)"
    }
]

# === ОПТИМИЗИРОВАННЫЙ ПРОМПТ (v2) ===

# NEW: Строгий system prompt на английском
SYSTEM_PROMPT = """You are a strict technical editor used for automated text processing.
Your ONLY goal is to output the edited text in Markdown format.
DO NOT converse. DO NOT use introduction or conclusion phrases.
DO NOT output internal reasoning or <think> tags.
Maintain 100% technical accuracy."""

def build_prompt(chapter_title: str, chapter_text: str) -> str:
    """
    Строит оптимизированный промпт (v2).
    Адаптирован для MoE-моделей (gpt-oss) и стандартных LLM (Qwen/Gemma).

    Источник: optimal-prompts-recomended.md
    """
    return f"""Ты — профессиональный технический редактор (Technical Writer).
Твоя задача — превратить сырую расшифровку вебинара в чистую, структурированную статью в формате Markdown.

**ВХОДНЫЕ ДАННЫЕ:**
Тема главы: "{chapter_title}"

**СТРОГИЕ ЗАПРЕТЫ (NEGATIVE CONSTRAINTS):**
1. ЗАПРЕЩЕНО писать вступления ("Конечно, вот текст...", "Analysis:..."). Сразу выдавай результат.
2. ЗАПРЕЩЕНО использовать теги мышления (<think>) в финальном ответе.
3. ЗАПРЕЩЕНО удалять примеры кода или менять логику лектора.

**ИНСТРУКЦИИ ПО ОБРАБОТКЕ:**

1. **ТЕРМИНОЛОГИЯ И КОД (КРИТИЧНО):**
   - Исправь фонетические ошибки распознавания (STT):
     • "ПыТорч" / "пайторч" -> `PyTorch`
     • "джипити" / "GPT-шка" -> `GPT`
     • "тензорфлоу" -> `TensorFlow`
     • "керас" -> `Keras`
     • "нампай" -> `NumPy`, "пандас" -> `Pandas`
     • "джупитер" -> `Jupyter`, "колаб" -> `Colab`
     • "хаггин фейс" -> `Hugging Face`
   - Все библиотеки, методы, переменные и пути к файлам выделяй как `код` (в обратных кавычках).
   - Ключевые термины при первом упоминании выделяй **жирным**.

2. **ЧИСТКА ТЕКСТА:**
   - УДАЛИ мусор: "поставьте плюсики", "слышно меня?", "э-э-э", "ну как бы".
   - УДАЛИ повторы (Looping): Если фраза повторяется 2+ раза подряд, оставь только одну версию.
   - УДАЛИ маркетинг: "успейте купить", "ссылка в описании".
   - УДАЛИ организационные моменты (перекличка, настройка звука).
   - СЖИМАЙ воду, но сохраняй технический смысл.

3. **СТРУКТУРА (MARKDOWN):**
   - Разбей текст на смысловые абзацы (не делай стены текста).
   - Используй заголовки уровня ## и ### для разделения подтем.
   - Используй маркированные списки (-) для перечислений.

**ПРИМЕР (ONE-SHOT):**
*Вход:* "Короче, импортируем нампай как нп. Эээ... потом делаем массив. Массив делаем."
*Выход:* "Сначала импортируем библиотеку `NumPy` как `np`. Затем создаем массив данных."

**ИСХОДНЫЙ ТЕКСТ:**
{chapter_text}

**ГОТОВАЯ СТАТЬЯ (MARKDOWN):**"""


def load_chapter_text(chapter_info: dict) -> tuple[str, str]:
    """Загружает текст главы из файлов."""
    chapters_file = Path(chapter_info["file"])
    packed_file = Path(chapter_info["packed"])

    with chapters_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chapter = data["chapters"][chapter_info["chapter_idx"]]

    # Загружаем абзацы
    paragraphs = {}
    with packed_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("type") == "paragraph":
                paragraphs[row["id"]] = row["text"]

    # Собираем текст
    chapter_text = "\n\n".join(
        paragraphs[pid] for pid in chapter["paragraph_ids"] if pid in paragraphs
    )

    return chapter["title"], chapter_text


def test_model(model_name: str, chapter_title: str, chapter_text: str) -> dict:
    """Тестирует одну модель на одной главе."""
    print(f"\n{'='*80}")
    print(f"Модель: {model_name}")
    print(f"Глава: {chapter_title}")
    print(f"Исходная длина: {len(chapter_text)} символов")

    prompt = build_prompt(chapter_title, chapter_text)

    start_time = time.time()

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
                "repeat_penalty": REPEAT_PENALTY,  # NEW: Предотвращает зацикливание
                "num_ctx": NUM_CTX,                # NEW: Больше контекста
            },
            think=False
        )

        edited_text = response['message']['content'].strip()
        elapsed_time = time.time() - start_time

        compression_ratio = len(edited_text) / len(chapter_text)

        print(f"[OK] Успешно обработано")
        print(f"  Время: {elapsed_time:.2f} сек")
        print(f"  Новая длина: {len(edited_text)} символов")
        print(f"  Компрессия: {compression_ratio:.2%} (удалено {(1-compression_ratio)*100:.1f}%)")

        return {
            "model": model_name,
            "chapter": chapter_title,
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
        print(f"[ERROR] ОШИБКА: {e}")

        return {
            "model": model_name,
            "chapter": chapter_title,
            "success": False,
            "time": round(elapsed_time, 2),
            "original_length": len(chapter_text),
            "edited_length": 0,
            "compression_ratio": 0,
            "edited_text": "",
            "error": str(e)
        }


def main():
    """Основная функция тестирования."""
    print("=== Сравнительное тестирование LLM-моделей (v2 - Optimized) ===")
    print(f"Моделей: {len(MODELS)}")
    print(f"Тестовых глав: {len(TEST_CHAPTERS)}")
    print(f"Всего тестов: {len(MODELS) * len(TEST_CHAPTERS)}")
    print(f"\nПАРАМЕТРЫ (v2):")
    print(f"  temperature: {TEMPERATURE}")
    print(f"  repeat_penalty: {REPEAT_PENALTY} (NEW!)")
    print(f"  num_ctx: {NUM_CTX} (NEW!)")
    print(f"  max_tokens: {MAX_TOKENS}")

    # Проверяем доступность моделей
    print("\n=== Проверка доступности моделей ===")
    available_models = ollama.list()
    model_names = [m.model for m in available_models.models]

    for model in MODELS:
        # Ollama может возвращать имена с тегами, проверяем частичное совпадение
        found = any(model in name or name.startswith(model.split(':')[0]) for name in model_names)
        status = "[OK]" if found else "[НЕ НАЙДЕНА]"
        print(f"  {model}: {status}")

    print("\n=== Начинаем тестирование ===\n")

    results = []

    # Запускаем тесты
    for chapter_info in TEST_CHAPTERS:
        print(f"\n{'#'*80}")
        print(f"# {chapter_info['name']}")
        print(f"{'#'*80}")

        chapter_title, chapter_text = load_chapter_text(chapter_info)

        for model in MODELS:
            result = test_model(model, chapter_title, chapter_text)
            results.append(result)

            # Сохраняем отредактированный текст для ручного сравнения
            if result["success"]:
                output_file = Path(f"test_results_v2/{model.replace(':', '_')}_{chapter_info['chapter_idx']}.txt")
                output_file.parent.mkdir(exist_ok=True)
                output_file.write_text(result["edited_text"], encoding="utf-8")

    # Сохраняем полный отчёт
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(f"test_results_v2/comparison_report_v2_{timestamp}.json")
    report_file.parent.mkdir(exist_ok=True)

    with report_file.open("w", encoding="utf-8") as f:
        json.dump({
            "version": "2.0-optimized",
            "timestamp": timestamp,
            "models": MODELS,
            "test_chapters": [c["name"] for c in TEST_CHAPTERS],
            "parameters": {
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "repeat_penalty": REPEAT_PENALTY,
                "num_ctx": NUM_CTX
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    # Печатаем итоговую таблицу
    print(f"\n{'='*80}")
    print("[v2] ИТОГОВАЯ ТАБЛИЦА")
    print(f"{'='*80}\n")

    # Группируем по моделям
    for model in MODELS:
        model_results = [r for r in results if r["model"] == model]
        successful = [r for r in model_results if r["success"]]

        if not successful:
            print(f"[FAILED] {model}: ВСЕ ТЕСТЫ ПРОВАЛЕНЫ")
            continue

        avg_time = sum(r["time"] for r in successful) / len(successful)
        avg_compression = sum(r["compression_ratio"] for r in successful) / len(successful)

        print(f"[OK] {model}")
        print(f"  Успешно: {len(successful)}/{len(model_results)}")
        print(f"  Средняя скорость: {avg_time:.2f} сек/глава")
        print(f"  Средняя компрессия: {avg_compression:.2%} (удалено {(1-avg_compression)*100:.1f}%)")
        print()

    print(f"\n[REPORT] Полный отчёт сохранён: {report_file}")
    print(f"[FILES] Отредактированные тексты: test_results_v2/")
    print("\n=== Тестирование v2 завершено! ===")


if __name__ == "__main__":
    main()
