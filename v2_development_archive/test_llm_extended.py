"""Расширенный тест LLM на большом куске текста."""

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

# LM Studio на порту 1234
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

def main():
    print("=" * 70)
    print("РАСШИРЕННЫЙ ТЕСТ: Большой кусок текста")
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

    # Читаем первые N абзацев
    NUM_PARAGRAPHS = 15  # Берём 15 абзацев
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
        print(f"Длительность фрагмента: ~{duration:.1f} секунд ({duration/60:.1f} минут)")

    print("\n" + "=" * 70)
    print("ПОЛНЫЙ ТЕКСТ ФРАГМЕНТА:")
    print("=" * 70)
    print(full_text)
    print("\n" + "=" * 70)

    # Тест 1: Присвоение названия
    print("\n[ТЕСТ 1] Присвоение названия главе")
    print("-" * 70)

    prompt1 = f"""Проанализируй следующий фрагмент видео-лекции и дай ему короткое название (3-7 слов).
Название должно отражать основную тему, о которой говорится в тексте.
Отвечай ТОЛЬКО названием, без кавычек, без дополнительных объяснений.

Текст:
{full_text}

Название:"""

    print("Отправляю запрос к LLM...")

    response1 = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "Ты эксперт по анализу образовательного контента. Создаёшь точные краткие названия для разделов лекций."},
            {"role": "user", "content": prompt1}
        ],
        temperature=0.3,
        max_tokens=50
    )

    title = response1.choices[0].message.content.strip()
    print(f"\n>>> НАЗВАНИЕ: {title}")

    # Тест 2: Краткое резюме
    print("\n" + "=" * 70)
    print("[ТЕСТ 2] Краткое резюме (2-3 предложения)")
    print("-" * 70)

    prompt2 = f"""Создай краткое резюме (2-3 предложения) следующего фрагмента лекции.
Резюме должно передать основную идею и ключевые моменты.

Текст:
{full_text}

Резюме:"""

    print("Отправляю запрос к LLM...")

    response2 = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "Ты эксперт по суммаризации образовательного контента. Создаёшь краткие информативные резюме."},
            {"role": "user", "content": prompt2}
        ],
        temperature=0.3,
        max_tokens=200
    )

    summary = response2.choices[0].message.content.strip()
    print(f"\n>>> РЕЗЮМЕ:\n{summary}")

    # Тест 3: Развёрнутое резюме с ключевыми пунктами
    print("\n" + "=" * 70)
    print("[ТЕСТ 3] Развёрнутое резюме с ключевыми пунктами")
    print("-" * 70)

    prompt3 = f"""Проанализируй следующий фрагмент лекции и создай структурированное резюме:

1. Основная тема (1 предложение)
2. Ключевые пункты (3-5 пунктов списком)
3. Упомянутые технологии/термины (если есть)

Текст:
{full_text}"""

    print("Отправляю запрос к LLM...")

    response3 = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "Ты эксперт по анализу образовательного контента. Создаёшь структурированные детальные резюме."},
            {"role": "user", "content": prompt3}
        ],
        temperature=0.3,
        max_tokens=400
    )

    detailed = response3.choices[0].message.content.strip()
    print(f"\n>>> СТРУКТУРИРОВАННОЕ РЕЗЮМЕ:\n{detailed}")

    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print(f"✓ Файл: {sample_file.name}")
    print(f"✓ Обработано абзацев: {len(paragraphs)}")
    print(f"✓ Длина текста: {len(full_text)} символов")
    print(f"✓ Все 3 теста выполнены успешно")
    print("\nМодель показывает:")
    print("  - Способность к присвоению названий")
    print("  - Качество суммаризации")
    print("  - Структурированный анализ контента")
    print()

if __name__ == "__main__":
    main()
