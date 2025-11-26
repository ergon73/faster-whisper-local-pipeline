"""Тест улучшенного промпта на проблемных главах.

Цель: Проверить, помогает ли добавление fallback-инструкции для плотного технического контента.
"""
import json
import time
import re
from pathlib import Path

try:
    import ollama
except ImportError:
    print("ERROR: pip install ollama")
    exit(1)

# === КОНФИГУРАЦИЯ ===

MODEL_NAME = "qwen3:32b"

SYSTEM_PROMPT = "You are a professional technical blog editor. Output clean Markdown only."

MODEL_OPTIONS = {
    "temperature": 0.3,
    "repeat_penalty": 1.1,
    "top_k": 40,
    "num_ctx": 8192,
    "num_predict": 3000
}

# === УЛУЧШЕННЫЙ ПРОМПТ ===

def build_prompt_enhanced(chapter_title: str, chapter_text: str) -> str:
    """Промпт с fallback-инструкцией для плотного технического контента."""
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
    """Пост-обработка с regex-очисткой (страховка)."""
    # 1. Удаляем <think> теги
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. Извлекаем из Markdown-обертки
    match = re.search(r'```(?:markdown)?\s*(.*?)```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    return text.strip()


# === ТЕСТОВЫЕ ГЛАВЫ ===

TEST_CHAPTERS = [
    # Проблемные главы (День 3, главы 8-12)
    {"day": "День 3", "chapter_idx": 7, "type": "ПРОБЛЕМНАЯ"},
    {"day": "День 3", "chapter_idx": 8, "type": "ПРОБЛЕМНАЯ"},
    {"day": "День 3", "chapter_idx": 9, "type": "ПРОБЛЕМНАЯ"},
    {"day": "День 3", "chapter_idx": 10, "type": "ПРОБЛЕМНАЯ"},
    {"day": "День 3", "chapter_idx": 11, "type": "ПРОБЛЕМНАЯ"},

    # Контрольные (успешные главы для проверки, что не сломали)
    {"day": "День 3", "chapter_idx": 0, "type": "КОНТРОЛЬ (успешная)"},
    {"day": "День 3", "chapter_idx": 5, "type": "КОНТРОЛЬ (успешная)"},
]

DAY_FILES = {
    "День 3": {
        "chapters": "transcribe/Большой марафон по Классическому AI. День 3_video_mp4_chapters.json",
        "packed": "transcribe/Большой марафон по Классическому AI. День 3_video_mp4_packed.jsonl"
    }
}


def load_chapter_data(day: str, chapter_idx: int):
    """Загружает данные главы."""
    files = DAY_FILES[day]

    with open(files["chapters"], "r", encoding="utf-8") as f:
        data = json.load(f)
    chapter = data["chapters"][chapter_idx]

    # Загружаем абзацы
    paragraphs = {}
    with open(files["packed"], "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("type") == "paragraph":
                paragraphs[row["id"]] = row["text"]

    # Собираем текст главы
    chapter_text = "\n\n".join(
        paragraphs[pid] for pid in chapter["paragraph_ids"] if pid in paragraphs
    )

    return chapter["title"], chapter_text


def test_chapter(day: str, chapter_idx: int, chapter_type: str):
    """Тестирует одну главу с новым промптом."""
    chapter_title, chapter_text = load_chapter_data(day, chapter_idx)

    print(f"\n{'='*80}")
    print(f"{chapter_type}: {day}, Глава {chapter_idx+1}")
    print(f"{'='*80}")
    print(f"Название: {chapter_title}")
    print(f"Исходная длина: {len(chapter_text)} символов")

    prompt = build_prompt_enhanced(chapter_title, chapter_text)
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
        edited_text = clean_output(edited_text)

        elapsed_time = time.time() - start_time
        compression_ratio = len(edited_text) / len(chapter_text)

        print(f"[OK] Время: {elapsed_time:.2f} сек")
        print(f"  Новая длина: {len(edited_text)} символов")
        print(f"  Компрессия: {compression_ratio:.2%}")

        # Сохраняем результат
        output_file = Path(f"enhanced_output/{day}_chapter_{chapter_idx+1}.md")
        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text(edited_text, encoding="utf-8")

        # Показываем первые 500 символов
        print(f"\n=== ПЕРВЫЕ 500 СИМВОЛОВ ===")
        print(edited_text[:500])

        return {
            "success": True,
            "time": elapsed_time,
            "original_length": len(chapter_text),
            "edited_length": len(edited_text),
            "compression_ratio": compression_ratio
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[ERROR] {e}")

        return {
            "success": False,
            "time": elapsed_time,
            "error": str(e)
        }


def main():
    """Основная функция."""
    print("=== ТЕСТ УЛУЧШЕННОГО ПРОМПТА ===")
    print(f"Модель: {MODEL_NAME}")
    print(f"Тестовых глав: {len(TEST_CHAPTERS)}")
    print(f"  - Проблемных: {sum(1 for t in TEST_CHAPTERS if 'ПРОБЛЕМ' in t['type'])}")
    print(f"  - Контрольных: {sum(1 for t in TEST_CHAPTERS if 'КОНТРОЛЬ' in t['type'])}")

    results = []

    for test_case in TEST_CHAPTERS:
        result = test_chapter(
            test_case["day"],
            test_case["chapter_idx"],
            test_case["type"]
        )
        result["type"] = test_case["type"]
        result["chapter_idx"] = test_case["chapter_idx"]
        results.append(result)

    # Итоговая статистика
    print(f"\n{'='*80}")
    print("ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*80}\n")

    problematic = [r for r in results if "ПРОБЛЕМ" in r["type"]]
    control = [r for r in results if "КОНТРОЛЬ" in r["type"]]

    print("ПРОБЛЕМНЫЕ ГЛАВЫ (старый промпт: 7-9% компрессия):")
    for r in problematic:
        if r["success"]:
            status = "✅" if r["compression_ratio"] > 0.15 else "⚠️"
            print(f"  {status} Глава {r['chapter_idx']+1}: {r['compression_ratio']:.2%} компрессия, {r['time']:.1f} сек")
        else:
            print(f"  ❌ Глава {r['chapter_idx']+1}: ОШИБКА")

    print("\nКОНТРОЛЬНЫЕ ГЛАВЫ (проверка, что не сломали):")
    for r in control:
        if r["success"]:
            status = "✅" if 0.20 < r["compression_ratio"] < 0.80 else "⚠️"
            print(f"  {status} Глава {r['chapter_idx']+1}: {r['compression_ratio']:.2%} компрессия, {r['time']:.1f} сек")
        else:
            print(f"  ❌ Глава {r['chapter_idx']+1}: ОШИБКА")

    # Сравнение с оригинальными результатами
    print(f"\n{'='*80}")
    print("СРАВНЕНИЕ С ОРИГИНАЛОМ:")
    print(f"{'='*80}")
    print("Старый промпт (проблемные главы 8-12): 7-9% компрессия")

    if problematic and all(r["success"] for r in problematic):
        avg_compression = sum(r["compression_ratio"] for r in problematic) / len(problematic)
        print(f"Новый промпт (проблемные главы 8-12): {avg_compression:.2%} компрессия")

        if avg_compression > 0.15:
            print("\n✅ УЛУЧШЕНИЕ! Модель генерирует больше контента.")
        else:
            print("\n⚠️ Проблема не решена. Нужна дополнительная настройка.")

    print(f"\n[OUTPUT] Результаты сохранены в: enhanced_output/")


if __name__ == "__main__":
    main()
