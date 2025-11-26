"""Тест LLM-редактирования одной главы для диагностики."""

import json
import sys
from pathlib import Path

try:
    import ollama
except ImportError:
    print("ERROR: pip install ollama")
    sys.exit(1)

# Читаем Главу 1 из День 1
chapters_file = Path("transcribe/Большой марафон по Классическому AI. День 1._video_mp4_chapters.json")
packed_file = Path("transcribe/Большой марафон по Классическому AI. День 1._video_mp4_packed.jsonl")

with chapters_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

chapter_1 = data["chapters"][0]  # Глава 1
print(f"Глава: {chapter_1['title']}")
print(f"Paragraph IDs: {chapter_1['paragraph_ids']}")

# Читаем абзацы
paragraphs = {}
with packed_file.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line.strip())
        if row.get("type") == "paragraph":
            paragraphs[row["id"]] = row["text"]

# Собираем текст главы
chapter_text = "\n\n".join(paragraphs[pid] for pid in chapter_1["paragraph_ids"] if pid in paragraphs)

print(f"\nИсходная длина: {len(chapter_text)} символов")
print(f"\nПервые 500 символов:\n{chapter_text[:500]}\n")

# Редактируем через LLM
prompt = f"""Ты технический редактор образовательного видео-контента.

**Задача:** улучшить текст транскрипта этой главы видео-лекции.

**ОБЯЗАТЕЛЬНО:**
✓ Исправь ошибки распознавания речи (например: "ПыТорч" → "PyTorch", "джипити" → "GPT")
✓ Убери речевые повторы и "воду" (избыточные вводные фразы)
✓ Сохрани ВСЕ технические детали, термины, названия технологий, примеры кода
✓ Сохрани ВСЕ смысловые идеи, концепции, объяснения
✓ Сделай текст связным и структурированным (логичные абзацы)
✓ Сохрани стиль устной речи лектора (не делай текст слишком формальным)

**МИНИМИЗИРУЙ или УБЕРИ:**
✗ Маркетинговые фразы ("лучший курс", "уникальная возможность", "успей купить")
✗ Призывы к действию не относящиеся к обучению ("подпишись", "поставь лайк")
✗ Избыточные приветствия и прощания (оставь только если это начало/конец)

**ОБЯЗАТЕЛЬНО СОХРАНИ:**
• Все технологии, библиотеки, инструменты, фреймворки
• Примеры команд, кода, конфигураций
• Объяснения концепций и алгоритмов
• Логику рассуждений и аргументацию
• Ссылки на ресурсы, документацию
• Практические советы и рекомендации
• Числа, метрики, результаты экспериментов

**Формат вывода:**
Связный текст, разбитый на абзацы. Не добавляй заголовков или разметки - только улучшенный текст.

**Исходный текст главы "{chapter_1['title']}":**

{chapter_text}

**Улучшенная версия:**"""

print("Отправляю запрос к LLM...")

response = ollama.chat(
    model="qwen3:32b",
    messages=[
        {
            "role": "system",
            "content": "Ты эксперт технический редактор образовательного контента. Ты специализируешься на улучшении транскриптов технических лекций."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0.3,
        "num_predict": 3000
    },
    think=False
)

edited_text = response['message']['content'].strip()

print(f"\nОтредактированная длина: {len(edited_text)} символов")
print(f"Компрессия: {len(edited_text) / len(chapter_text):.2%}")
print(f"\n{'='*80}")
print("ИСХОДНЫЙ ТЕКСТ:")
print(f"{'='*80}\n")
print(chapter_text)
print(f"\n{'='*80}")
print("ОТРЕДАКТИРОВАННЫЙ ТЕКСТ:")
print(f"{'='*80}\n")
print(edited_text)
print(f"\n{'='*80}")

# Сохраняем для сравнения
with open("test_chapter1_original.txt", "w", encoding="utf-8") as f:
    f.write(chapter_text)

with open("test_chapter1_edited.txt", "w", encoding="utf-8") as f:
    f.write(edited_text)

print("\n✓ Сохранено в test_chapter1_original.txt и test_chapter1_edited.txt")
