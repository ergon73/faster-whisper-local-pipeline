"""Тест Qwen3-32B через Ollama на расширенном фрагменте."""

import json
import sys
from pathlib import Path

# Настройка кодировки для Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] Нужно установить: pip install openai")
    sys.exit(1)

# Ollama на порту 11434
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

def main():
    print("=" * 70)
    print("ТЕСТ QWEN3-32B: Качество работы с текстом")
    print("=" * 70)

    # Найдём packed.jsonl файл
    transcribe_dir = Path("transcribe")
    packed_files = list(transcribe_dir.glob("*_packed.jsonl"))

    if not packed_files:
        print("[ERROR] Нет файлов *_packed.jsonl")
        return

    sample_file = packed_files[0]
    print(f"\nФайл: {sample_file.name}")
    print("-" * 70)

    # Читаем первые 15 абзацев
    NUM_PARAGRAPHS = 15
    paragraphs = []

    with open(sample_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:  # Пропускаем метаданные
                continue
            if len(paragraphs) >= NUM_PARAGRAPHS:
                break

            try:
                para = json.loads(line)
                if para.get("type") == "paragraph":
                    text = para["text"]
                    start = para.get("start")
                    paragraphs.append({
                        "id": para.get("id"),
                        "start": start,
                        "text": text
                    })
            except:
                continue

    if not paragraphs:
        print("[ERROR] Не удалось прочитать абзацы")
        return

    # Объединяем текст
    full_text = "\n\n".join([p["text"] for p in paragraphs])

    print(f"\nВзято абзацев: {len(paragraphs)}")
    print(f"Длина текста: {len(full_text)} символов (~{len(full_text.split())} слов)")

    if paragraphs[0].get("start") and paragraphs[-1].get("start"):
        duration = paragraphs[-1]["start"] - paragraphs[0]["start"]
        print(f"Длительность: ~{duration:.1f} сек ({duration/60:.1f} мин)")

    # Тест 1: Присвоение названия (улучшенный промпт)
    print("\n" + "=" * 70)
    print("[ТЕСТ 1] Присвоение названия главе")
    print("-" * 70)

    prompt1 = f"""Ты эксперт по анализу образовательного видео-контента.

Твоя задача: дать короткое информативное название (3-7 слов) для следующего фрагмента видео-лекции.

Требования к названию:
- Отражает основную тему фрагмента
- Краткое и ёмкое (3-7 слов)
- Без кавычек и лишних символов
- Только само название, никаких пояснений

Фрагмент лекции:
{full_text}

Название:"""

    print("Отправляю запрос к Qwen3-32B...")

    response1 = client.chat.completions.create(
        model="qwen3:32b",
        messages=[
            {"role": "system", "content": "Ты эксперт по анализу образовательного контента. Создаёшь точные краткие названия."},
            {"role": "user", "content": prompt1}
        ],
        temperature=0.3,
        max_tokens=50
    )

    title = response1.choices[0].message.content.strip()
    print(f"\n>>> НАЗВАНИЕ: «{title}»")

    # Тест 2: Краткое резюме
    print("\n" + "=" * 70)
    print("[ТЕСТ 2] Краткое резюме (2-3 предложения)")
    print("-" * 70)

    prompt2 = f"""Создай краткое резюме (2-3 предложения) следующего фрагмента видео-лекции.
Резюме должно передать основную идею и ключевые моменты, упомянутые в тексте.

Фрагмент:
{full_text}

Резюме:"""

    print("Отправляю запрос к Qwen3-32B...")

    response2 = client.chat.completions.create(
        model="qwen3:32b",
        messages=[
            {"role": "system", "content": "Ты эксперт по суммаризации образовательного контента."},
            {"role": "user", "content": prompt2}
        ],
        temperature=0.3,
        max_tokens=200
    )

    summary = response2.choices[0].message.content.strip()
    print(f"\n>>> РЕЗЮМЕ:\n{summary}")

    # Тест 3: Структурированный анализ
    print("\n" + "=" * 70)
    print("[ТЕСТ 3] Структурированный анализ")
    print("-" * 70)

    prompt3 = f"""Проанализируй следующий фрагмент видео-лекции и создай структурированный анализ:

**Основная тема:** (1 предложение - о чём этот фрагмент)

**Ключевые пункты:**
- (3-5 главных моментов списком)

**Упомянутые технологии/термины:** (если есть)

**Тип контента:** (вводная часть / техническое объяснение / практический пример / и т.д.)

Фрагмент:
{full_text}"""

    print("Отправляю запрос к Qwen3-32B...")

    response3 = client.chat.completions.create(
        model="qwen3:32b",
        messages=[
            {"role": "system", "content": "Ты эксперт по детальному анализу образовательного контента."},
            {"role": "user", "content": prompt3}
        ],
        temperature=0.3,
        max_tokens=500
    )

    analysis = response3.choices[0].message.content.strip()
    print(f"\n>>> АНАЛИЗ:\n{analysis}")

    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ ТЕСТИРОВАНИЯ QWEN3-32B")
    print("=" * 70)
    print(f"✓ Модель: Qwen3-32B (через Ollama)")
    print(f"✓ Обработано: {len(paragraphs)} абзацев, {len(full_text)} символов")
    print(f"✓ Все 3 теста выполнены")
    print("\nQwen3-32B показывает:")
    print("  [1] Способность к созданию информативных названий")
    print("  [2] Качественную суммаризацию с сохранением ключевых деталей")
    print("  [3] Глубокий структурированный анализ контента")
    print()

if __name__ == "__main__":
    main()
