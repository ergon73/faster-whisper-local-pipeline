"""Тест Qwen3-32B с отключенным thinking mode (нативный Ollama API)."""

import json
import sys
from pathlib import Path

# Настройка кодировки для Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    import ollama
except ImportError:
    print("[ERROR] Установите: pip install ollama")
    sys.exit(1)

def main():
    print("=" * 70)
    print("ТЕСТ QWEN3-32B: Отключен Thinking Mode")
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
            if i == 0:
                continue
            if len(paragraphs) >= NUM_PARAGRAPHS:
                break

            try:
                para = json.loads(line)
                if para.get("type") == "paragraph":
                    paragraphs.append({
                        "id": para.get("id"),
                        "start": para.get("start"),
                        "text": para["text"]
                    })
            except:
                continue

    if not paragraphs:
        print("[ERROR] Не удалось прочитать абзацы")
        return

    full_text = "\n\n".join([p["text"] for p in paragraphs])

    print(f"\nВзято абзацев: {len(paragraphs)}")
    print(f"Длина текста: {len(full_text)} символов (~{len(full_text.split())} слов)")

    # Тест 1: Название главы (Few-Shot с отключенным thinking)
    print("\n" + "=" * 70)
    print("[ТЕСТ 1] Присвоение названия главе")
    print("-" * 70)

    messages1 = [
        {"role": "system", "content": "Ты эксперт по анализу образовательного видео. Давай короткие точные названия (3-7 слов) для фрагментов лекций."},

        # Few-shot примеры
        {"role": "user", "content": "Дай короткое название (3-7 слов):\n\nСегодня мы разберём основы линейной регрессии. Это базовый алгоритм машинного обучения."},
        {"role": "assistant", "content": "Основы линейной регрессии"},

        {"role": "user", "content": "Дай короткое название (3-7 слов):\n\nВ этой части установим PyTorch и создадим первую нейронную сеть."},
        {"role": "assistant", "content": "Установка PyTorch и первая нейросеть"},

        # Реальный запрос
        {"role": "user", "content": f"Дай короткое название (3-7 слов):\n\n{full_text[:1000]}..."}
    ]

    print("Отправляю запрос к Qwen3-32B (think=False)...")

    response1 = ollama.chat(
        model="qwen3:32b",
        messages=messages1,
        options={"temperature": 0.3, "num_predict": 50},
        think=False  # ОТКЛЮЧАЕМ THINKING MODE
    )

    title = response1['message']['content'].strip()
    print(f"\n>>> НАЗВАНИЕ: «{title}»")

    # Тест 2: Краткое резюме
    print("\n" + "=" * 70)
    print("[ТЕСТ 2] Краткое резюме")
    print("-" * 70)

    messages2 = [
        {"role": "system", "content": "Ты эксперт по суммаризации. Создаёшь краткие резюме из 2-3 предложений."},
        {"role": "user", "content": f"Создай краткое резюме (2-3 предложения) этого фрагмента:\n\n{full_text}"}
    ]

    print("Отправляю запрос...")

    response2 = ollama.chat(
        model="qwen3:32b",
        messages=messages2,
        options={"temperature": 0.3, "num_predict": 200},
        think=False
    )

    summary = response2['message']['content'].strip()
    print(f"\n>>> РЕЗЮМЕ:\n{summary}")

    # Тест 3: Извлечение ключевых тем
    print("\n" + "=" * 70)
    print("[ТЕСТ 3] Ключевые темы и технологии")
    print("-" * 70)

    messages3 = [
        {"role": "system", "content": "Ты эксперт по анализу образовательного контента."},
        {"role": "user", "content": f"""Извлеки из фрагмента:

1. Главные темы (2-4 пункта)
2. Упомянутые технологии/библиотеки
3. Ключевые концепции

Используй маркированные списки.

Фрагмент:
{full_text}"""}
    ]

    print("Отправляю запрос...")

    response3 = ollama.chat(
        model="qwen3:32b",
        messages=messages3,
        options={"temperature": 0.3, "num_predict": 400},
        think=False
    )

    analysis = response3['message']['content'].strip()
    print(f"\n>>> АНАЛИЗ:\n{analysis}")

    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ")
    print("=" * 70)
    print("✓ Использован нативный Ollama API")
    print("✓ Thinking mode отключен (think=False)")
    print("✓ Все тесты выполнены успешно")
    print("\nРезультат:")
    print(f"  Название: {len(title)} символов")
    print(f"  Резюме: {len(summary)} символов")
    print(f"  Анализ: {len(analysis)} символов")
    print()

if __name__ == "__main__":
    main()
