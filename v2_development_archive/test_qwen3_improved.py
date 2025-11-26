"""Тест Qwen3-32B с улучшенными промптами (few-shot)."""

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
    print("УЛУЧШЕННЫЙ ТЕСТ QWEN3-32B: Few-Shot промпты")
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

    if paragraphs[0].get("start") and paragraphs[-1].get("start"):
        duration = paragraphs[-1]["start"] - paragraphs[0]["start"]
        print(f"Длительность: ~{duration:.1f} сек ({duration/60:.1f} мин)")

    # Тест 1: Few-shot для названий
    print("\n" + "=" * 70)
    print("[ТЕСТ 1] Присвоение названия (Few-Shot)")
    print("-" * 70)

    # Few-shot промпт с примерами
    messages1 = [
        {"role": "system", "content": "Ты эксперт по анализу образовательного видео. Твоя задача - давать короткие точные названия (3-7 слов) для фрагментов лекций."},

        # Пример 1
        {"role": "user", "content": """Дай короткое название (3-7 слов) для этого фрагмента:

Сегодня мы разберём основы линейной регрессии. Это один из базовых алгоритмов машинного обучения. Мы посмотрим как работает метод наименьших квадратов и как оптимизировать параметры модели с помощью градиентного спуска."""},
        {"role": "assistant", "content": "Основы линейной регрессии и градиентный спуск"},

        # Пример 2
        {"role": "user", "content": """Дай короткое название (3-7 слов) для этого фрагмента:

В этой части мы установим PyTorch и создадим нашу первую нейронную сеть. Я покажу как определить архитектуру, задать функцию потерь и запустить обучение на реальных данных."""},
        {"role": "assistant", "content": "Установка PyTorch и первая нейросеть"},

        # Пример 3
        {"role": "user", "content": """Дай короткое название (3-7 слов) для этого фрагмента:

Прежде чем начать основную тему, хочу рассказать про интересную новость. Компания OpenAI выпустила новую модель GPT-4 Turbo с расширенным контекстом. Это открывает новые возможности для разработчиков."""},
        {"role": "assistant", "content": "Новости: релиз GPT-4 Turbo"},

        # Реальный запрос
        {"role": "user", "content": f"""Дай короткое название (3-7 слов) для этого фрагмента:

{full_text}"""}
    ]

    print("Отправляю few-shot запрос к Qwen3-32B...")

    response1 = client.chat.completions.create(
        model="qwen3:32b",
        messages=messages1,
        temperature=0.3,
        max_tokens=50
    )

    title = response1.choices[0].message.content.strip()
    print(f"\n>>> НАЗВАНИЕ: «{title}»")

    # Тест 2: Улучшенный промпт для резюме
    print("\n" + "=" * 70)
    print("[ТЕСТ 2] Краткое резюме (улучшенный промпт)")
    print("-" * 70)

    messages2 = [
        {"role": "system", "content": "Ты эксперт по суммаризации образовательного контента. Создаёшь краткие информативные резюме из 2-3 предложений."},

        # Пример
        {"role": "user", "content": """Создай краткое резюме (2-3 предложения):

Добрый день! Сегодня мы начнём изучение компьютерного зрения. Первая тема - свёрточные нейронные сети или CNN. Мы разберём как работают свёртки, пулинг и почему эта архитектура так эффективна для работы с изображениями. Покажу примеры на PyTorch."""},
        {"role": "assistant", "content": "Лекция посвящена основам компьютерного зрения и свёрточным нейронным сетям (CNN). Разбираются ключевые компоненты архитектуры: свёртки и пулинг, объясняется их эффективность для обработки изображений. Приводятся практические примеры реализации на PyTorch."},

        # Реальный запрос
        {"role": "user", "content": f"""Создай краткое резюме (2-3 предложения):

{full_text}"""}
    ]

    print("Отправляю запрос к Qwen3-32B...")

    response2 = client.chat.completions.create(
        model="qwen3:32b",
        messages=messages2,
        temperature=0.3,
        max_tokens=200
    )

    summary = response2.choices[0].message.content.strip()
    print(f"\n>>> РЕЗЮМЕ:\n{summary}")

    # Тест 3: Извлечение ключевых тем (для Этапа 3)
    print("\n" + "=" * 70)
    print("[ТЕСТ 3] Извлечение ключевых тем и терминов")
    print("-" * 70)

    messages3 = [
        {"role": "system", "content": "Ты эксперт по извлечению ключевой информации из образовательного контента."},
        {"role": "user", "content": f"""Проанализируй фрагмент лекции и извлеки:

1. **Главные темы** (2-4 темы списком)
2. **Технологии/библиотеки/инструменты** (если упоминаются)
3. **Ключевые концепции** (термины, алгоритмы, методы)

Будь конкретен и лаконичен. Используй маркированные списки.

Фрагмент:
{full_text}

Анализ:"""}
    ]

    print("Отправляю запрос к Qwen3-32B...")

    response3 = client.chat.completions.create(
        model="qwen3:32b",
        messages=messages3,
        temperature=0.3,
        max_tokens=400
    )

    analysis = response3.choices[0].message.content.strip()
    print(f"\n>>> ИЗВЛЕЧЁННАЯ ИНФОРМАЦИЯ:\n{analysis}")

    # Тест 4: Определение типа контента
    print("\n" + "=" * 70)
    print("[ТЕСТ 4] Классификация типа контента")
    print("-" * 70)

    messages4 = [
        {"role": "system", "content": "Ты эксперт по анализу структуры образовательного контента."},
        {"role": "user", "content": f"""Определи тип этого фрагмента лекции. Выбери один или несколько:

- Вводная часть / приветствие
- Анонс программы / план
- Теоретическое объяснение
- Практический пример / демонстрация
- Новости / актуальная информация
- Вопросы-ответы
- Заключение / выводы

Фрагмент:
{full_text}

Тип контента:"""}
    ]

    print("Отправляю запрос к Qwen3-32B...")

    response4 = client.chat.completions.create(
        model="qwen3:32b",
        messages=messages4,
        temperature=0.3,
        max_tokens=100
    )

    content_type = response4.choices[0].message.content.strip()
    print(f"\n>>> ТИП КОНТЕНТА:\n{content_type}")

    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ: Сравнение с базовым тестом")
    print("=" * 70)
    print("Улучшения few-shot промптов:")
    print("  [1] Название главы - примеры формата помогают модели")
    print("  [2] Резюме - чёткая структура запроса")
    print("  [3] Извлечение тем - конкретные инструкции")
    print("  [4] Классификация - список вариантов для выбора")
    print("\nQwen3-32B лучше работает когда:")
    print("  ✓ Есть примеры (few-shot)")
    print("  ✓ Чёткая структура запроса")
    print("  ✓ Конкретные инструкции")
    print()

if __name__ == "__main__":
    main()
