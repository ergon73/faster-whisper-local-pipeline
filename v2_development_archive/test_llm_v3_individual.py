"""Сравнительное тестирование LLM-моделей с индивидуальными промптами.

VERSION: 3.0 - Individual Optimized Prompts
DATE: 2025-11-26

ИЗМЕНЕНИЯ от v2:
1. Каждая модель имеет ИНДИВИДУАЛЬНЫЙ промпт (3 типа: Balanced, Verbose, Structured)
2. Каждая модель имеет ИНДИВИДУАЛЬНЫЕ параметры генерации
3. Учтена специфика каждой модели ("психотип"):
   - Qwen3:32b & Gemma3: Balanced (редактор Хабра)
   - GPT-OSS:20b: Verbose (корректор, НЕ СОКРАЩАЙ)
   - Qwen3:30b-a3b: Structured (XML-теги, разрешаем думать)

ИСТОЧНИКИ:
- optimal-prompts-recommended-v2.md
- optimal-prompts-recommended-v2-settings.md
"""

import json
import time
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable

try:
    import ollama
except ImportError:
    print("ERROR: pip install ollama")
    exit(1)

# === Конфигурация моделей ===

@dataclass
class ModelConfig:
    """Конфигурация для каждой модели."""
    name: str
    prompt_builder: Callable[[str, str], str]
    system_prompt: str
    options: dict
    description: str

# === ИНДИВИДУАЛЬНЫЕ ПРОМПТЫ ===

# 1. Balanced Prompt (Qwen3:32b, Gemma3)
def build_prompt_balanced(chapter_title: str, chapter_text: str) -> str:
    """Промпт для сбалансированных моделей (Qwen, Gemma)."""
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

**ВХОДНОЙ ТЕКСТ:**
Тема: {chapter_title}
Текст:
{chapter_text}

**СТАТЬЯ (MARKDOWN):**"""

# 2. Verbose Prompt (GPT-OSS:20b)
def build_prompt_verbose(chapter_title: str, chapter_text: str) -> str:
    """Промпт для GPT-OSS (против сокращения)."""
    return f"""Роль: Технический корректор (Technical Copywriter).
Задача: Исправить ошибки в транскрипте, сохранив **100% информативности** исходной лекции.

**ГЛАВНОЕ ПРАВИЛО:**
НЕ СОКРАЩАЙ ТЕКСТ. Твоя задача — не сделать саммари, а сделать читаемый полный текст лекции. Удаляй только слова-паразиты, но сохраняй все предложения, примеры и пояснения лектора.

**ИНСТРУКЦИИ:**
1. **Исправление ошибок:**
   - "ПыТорч" -> `PyTorch`, "джипити" -> `GPT`, "керас" -> `Keras`.
   - Исправь пунктуацию и разбей "поток сознания" на предложения.

2. **Оформление:**
   - Используй Markdown заголовки (##) для смены темы разговора.
   - Выделяй ключевые понятия **жирным**.
   - Код и команды пиши в `моноширинном формате`.

3. **Запреты:**
   - ЗАПРЕЩЕНО удалять технические детали, даже мелкие.
   - ЗАПРЕЩЕНО писать вступления ("Here is the text...").
   - ЗАПРЕЩЕНО объединять 5 предложений в одно. Пиши подробно.

**ИСХОДНЫЙ ТЕКСТ:**
Тема: {chapter_title}
Текст:
{chapter_text}

**ПОЛНЫЙ ОТРЕДАКТИРОВАННЫЙ ТЕКСТ:**"""

# 3. Structured Prompt (Qwen3:30b-a3b MoE)
def build_prompt_structured(chapter_title: str, chapter_text: str) -> str:
    """Промпт для MoE моделей (XML-теги)."""
    return f"""Ты — система обработки транскриптов.

Твоя задача состоит из двух этапов:
1. (Мысленно) Проанализировать текст на ошибки терминологии.
2. (Вывод) Написать чистый Markdown-текст.

**СЛОВАРЬ ЗАМЕН (Применять строго):**
- `PyTorch` (вместо ПыТорч/пайторч)
- `GPT` (вместо джипити)
- `TensorFlow`, `Keras`, `NumPy`, `Pandas`

**ФОРМАТ ВЫВОДА:**
Ты должен вывести ТОЛЬКО отредактированный текст внутри тегов <article> и </article>.
Никаких вступлений. Никаких объяснений после текста.

Пример:
<article>
## Заголовок
Текст лекции с исправленными терминами...
</article>

**ВХОДНЫЕ ДАННЫЕ:**
Тема: {chapter_title}
{chapter_text}

**ОТВЕТ:**"""

# === КОНФИГУРАЦИИ МОДЕЛЕЙ ===

MODEL_CONFIGS = {
    "qwen3:32b": ModelConfig(
        name="qwen3:32b",
        prompt_builder=build_prompt_balanced,
        system_prompt="You are a technical blog editor. Output clean Markdown only.",
        options={
            "temperature": 0.3,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "num_ctx": 8192,
            "num_predict": 3000
        },
        description="Qwen3 32B - Сбалансированный (Balanced)"
    ),

    "qwen3:30b-a3b": ModelConfig(
        name="qwen3:30b-a3b",
        prompt_builder=build_prompt_structured,
        system_prompt="You are a strict text processing system. Output only within <article> tags.",
        options={
            "temperature": 0.1,       # СТРОГО! Против рассуждений
            "repeat_penalty": 1.25,   # ЖЕСТКО! Против зацикливания
            "top_p": 0.9,
            "num_ctx": 8192,
            "num_predict": 3000
        },
        description="Qwen3 30B MoE - Структурированный (Structured, XML-теги)"
    ),

    "gpt-oss:20b": ModelConfig(
        name="gpt-oss:20b",
        prompt_builder=build_prompt_verbose,
        system_prompt="You are a technical copywriter. DO NOT summarize. Keep full content.",
        options={
            "temperature": 0.2,       # Низкая - четко следовать "НЕ СОКРАЩАЙ"
            "repeat_penalty": 1.05,   # МИНИМАЛЬНО! Разрешаем быть многословным
            "num_ctx": 8192,
            "num_predict": 4000       # Больше места для полного текста
        },
        description="GPT-OSS 20B - Многословный (Verbose, против сокращения)"
    ),

    "gemma3:27b-it-qat": ModelConfig(
        name="gemma3:27b-it-qat",
        prompt_builder=build_prompt_balanced,
        system_prompt="You are a technical blog editor. Output clean Markdown only.",
        options={
            "temperature": 0.3,
            "repeat_penalty": 1.15,   # Чуть выше - против тавтологии
            "num_ctx": 8192,
            "num_predict": 3000
        },
        description="Gemma3 27B IT-QAT - Сбалансированный (Balanced)"
    )
}

# === Тестовые главы ===

TEST_CHAPTERS = [
    {
        "file": "transcribe/Большой марафон по Классическому AI. День 1._video_mp4_chapters.json",
        "packed": "transcribe/Большой марафон по Классическому AI. День 1._video_mp4_packed.jsonl",
        "chapter_idx": 0,
        "name": "День 1, Глава 1 (длинная, ~7800 символов)"
    },
    {
        "file": "transcribe/Большой марафон по Классическому AI. День 2._video_mp4_chapters.json",
        "packed": "transcribe/Большой марафон по Классическому AI. День 2._video_mp4_packed.jsonl",
        "chapter_idx": 5,
        "name": "День 2, Глава 6 (средняя, ~3500 символов)"
    },
    {
        "file": "transcribe/Большой марафон по Классическому AI. День 3_video_mp4_chapters.json",
        "packed": "transcribe/Большой марафон по Классическому AI. День 3_video_mp4_packed.jsonl",
        "chapter_idx": 2,
        "name": "День 3, Глава 3 (короткая, ~2000 символов)"
    }
]

# === Функции ===

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


def extract_article_content(text: str) -> str:
    """Извлекает текст из <article> тегов (для Qwen MoE)."""
    match = re.search(r'<article>(.*?)</article>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Если тегов нет, возвращаем как есть


def test_model(model_name: str, chapter_title: str, chapter_text: str, config: ModelConfig) -> dict:
    """Тестирует одну модель на одной главе."""
    print(f"\n{'='*80}")
    print(f"Модель: {model_name}")
    print(f"Стратегия: {config.description}")
    print(f"Глава: {chapter_title}")
    print(f"Исходная длина: {len(chapter_text)} символов")

    prompt = config.prompt_builder(chapter_title, chapter_text)

    start_time = time.time()

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt}
            ],
            options=config.options,
            think=False
        )

        edited_text = response['message']['content'].strip()

        # Для Qwen MoE извлекаем текст из <article> тегов
        if model_name == "qwen3:30b-a3b":
            edited_text = extract_article_content(edited_text)

        elapsed_time = time.time() - start_time
        compression_ratio = len(edited_text) / len(chapter_text)

        print(f"[OK] Успешно обработано")
        print(f"  Время: {elapsed_time:.2f} сек")
        print(f"  Новая длина: {len(edited_text)} символов")
        print(f"  Компрессия: {compression_ratio:.2%} (удалено {(1-compression_ratio)*100:.1f}%)")

        return {
            "model": model_name,
            "strategy": config.description,
            "chapter": chapter_title,
            "success": True,
            "time": round(elapsed_time, 2),
            "original_length": len(chapter_text),
            "edited_length": len(edited_text),
            "compression_ratio": round(compression_ratio, 4),
            "edited_text": edited_text,
            "config": config.options,
            "error": None
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[ERROR] ОШИБКА: {e}")

        return {
            "model": model_name,
            "strategy": config.description,
            "chapter": chapter_title,
            "success": False,
            "time": round(elapsed_time, 2),
            "original_length": len(chapter_text),
            "edited_length": 0,
            "compression_ratio": 0,
            "edited_text": "",
            "config": config.options,
            "error": str(e)
        }


def main():
    """Основная функция тестирования."""
    models = list(MODEL_CONFIGS.keys())

    print("=== Сравнительное тестирование LLM (v3 - Individual Prompts) ===")
    print(f"Моделей: {len(models)}")
    print(f"Тестовых глав: {len(TEST_CHAPTERS)}")
    print(f"Всего тестов: {len(models) * len(TEST_CHAPTERS)}")

    print("\n=== КОНФИГУРАЦИИ МОДЕЛЕЙ ===")
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{model_name}:")
        print(f"  Стратегия: {config.description}")
        print(f"  Параметры: temp={config.options.get('temperature')}, "
              f"repeat_penalty={config.options.get('repeat_penalty')}")

    # Проверяем доступность моделей
    print("\n=== Проверка доступности моделей ===")
    available_models = ollama.list()
    model_names = [m.model for m in available_models.models]

    for model in models:
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

        for model_name in models:
            config = MODEL_CONFIGS[model_name]
            result = test_model(model_name, chapter_title, chapter_text, config)
            results.append(result)

            # Сохраняем отредактированный текст
            if result["success"]:
                output_file = Path(f"test_results_v3/{model_name.replace(':', '_')}_{chapter_info['chapter_idx']}.txt")
                output_file.parent.mkdir(exist_ok=True)
                output_file.write_text(result["edited_text"], encoding="utf-8")

    # Сохраняем полный отчёт
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(f"test_results_v3/comparison_report_v3_{timestamp}.json")
    report_file.parent.mkdir(exist_ok=True)

    with report_file.open("w", encoding="utf-8") as f:
        json.dump({
            "version": "3.0-individual",
            "timestamp": timestamp,
            "models": models,
            "test_chapters": [c["name"] for c in TEST_CHAPTERS],
            "configurations": {name: {
                "strategy": cfg.description,
                "options": cfg.options
            } for name, cfg in MODEL_CONFIGS.items()},
            "results": results
        }, f, indent=2, ensure_ascii=False)

    # Печатаем итоговую таблицу
    print(f"\n{'='*80}")
    print("[v3] ИТОГОВАЯ ТАБЛИЦА (Индивидуальные промпты)")
    print(f"{'='*80}\n")

    for model_name in models:
        model_results = [r for r in results if r["model"] == model_name]
        successful = [r for r in model_results if r["success"]]

        if not successful:
            print(f"[FAILED] {model_name}: ВСЕ ТЕСТЫ ПРОВАЛЕНЫ")
            continue

        avg_time = sum(r["time"] for r in successful) / len(successful)
        avg_compression = sum(r["compression_ratio"] for r in successful) / len(successful)

        print(f"[OK] {model_name}")
        print(f"  Стратегия: {MODEL_CONFIGS[model_name].description}")
        print(f"  Успешно: {len(successful)}/{len(model_results)}")
        print(f"  Средняя скорость: {avg_time:.2f} сек/глава")
        print(f"  Средняя компрессия: {avg_compression:.2%} (удалено {(1-avg_compression)*100:.1f}%)")
        print()

    print(f"\n[REPORT] Полный отчёт сохранён: {report_file}")
    print(f"[FILES] Отредактированные тексты: test_results_v3/")
    print("\n=== Тестирование v3 завершено! ===")


if __name__ == "__main__":
    main()
