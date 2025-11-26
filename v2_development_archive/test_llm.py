"""Тест интеграции с локальной LLM через LM Studio."""

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

# LM Studio обычно на порту 1234
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # LM Studio не требует ключ
)

def test_simple():
    """Простой тест работоспособности."""
    print("🧪 Тест 1: Простой запрос")
    print("-" * 50)

    response = client.chat.completions.create(
        model="local-model",  # LM Studio игнорирует это поле
        messages=[
            {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко."},
            {"role": "user", "content": "Скажи привет и назови себя в одном предложении."}
        ],
        temperature=0.3,
        max_tokens=100
    )

    answer = response.choices[0].message.content
    print(f"Ответ: {answer}\n")
    return True

def test_chapter_naming():
    """Тест присвоения названия главе из реальных данных."""
    print("🧪 Тест 2: Присвоение названия главе")
    print("-" * 50)

    # Найдём первый packed.jsonl файл
    transcribe_dir = Path("transcribe")
    if not transcribe_dir.exists():
        print("⚠️  Директория transcribe/ не найдена, пропускаем тест\n")
        return False

    packed_files = list(transcribe_dir.glob("*_packed.jsonl"))
    if not packed_files:
        print("⚠️  Нет файлов *_packed.jsonl, пропускаем тест\n")
        return False

    # Читаем первые 3 абзаца как пример главы
    sample_file = packed_files[0]
    print(f"📄 Используем: {sample_file.name}")

    paragraphs = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:  # Пропускаем метаданные
                continue
            if i > 3:  # Берём только 3 абзаца для теста
                break
            try:
                para = json.loads(line)
                if para.get("type") == "paragraph":
                    paragraphs.append(para["text"])
            except:
                continue

    if not paragraphs:
        print("⚠️  Не удалось прочитать абзацы\n")
        return False

    chapter_text = "\n\n".join(paragraphs)
    print(f"\n📝 Текст главы ({len(chapter_text)} символов):")
    print(chapter_text[:300] + "..." if len(chapter_text) > 300 else chapter_text)
    print()

    # Запрос к LLM
    prompt = f"""Дай короткое название (3-7 слов) для этой главы видео-лекции.
Название должно отражать основную тему обсуждаемую в тексте.
Отвечай ТОЛЬКО названием, без кавычек и дополнительных пояснений.

Текст главы:
{chapter_text}"""

    print("🤖 Отправляем запрос к LLM...")

    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "Ты эксперт по анализу текста. Твоя задача - давать краткие точные названия для разделов текста."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=50
    )

    title = response.choices[0].message.content.strip()
    print(f"\n✨ Предложенное название: «{title}»\n")
    return True

def test_summarization():
    """Тест суммаризации фрагмента."""
    print("🧪 Тест 3: Суммаризация текста")
    print("-" * 50)

    sample_text = """
    В этой лекции мы разбираем основы классического машинного обучения.
    Начнём с линейной регрессии, которая является базовым алгоритмом для предсказания непрерывных значений.
    Затем перейдём к логистической регрессии для задач классификации.
    Важно понимать функцию потерь и градиентный спуск для оптимизации параметров модели.
    В конце покажем практические примеры на Python с использованием библиотеки scikit-learn.
    """

    print(f"📝 Исходный текст:\n{sample_text.strip()}\n")

    response = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "Ты эксперт по суммаризации. Создавай краткие содержательные резюме."},
            {"role": "user", "content": f"Создай краткое резюме (1-2 предложения):\n\n{sample_text}"}
        ],
        temperature=0.3,
        max_tokens=150
    )

    summary = response.choices[0].message.content.strip()
    print(f"✨ Резюме: {summary}\n")
    return True

def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 50)
    print("🚀 Тестирование LLM интеграции")
    print("=" * 50 + "\n")

    try:
        # Проверяем доступность сервера
        print("🔍 Проверка подключения к LM Studio...")
        models = client.models.list()
        print(f"✅ Подключение успешно! Доступные модели: {len(models.data)}")
        if models.data:
            print(f"   Модель: {models.data[0].id}\n")

        # Запускаем тесты
        results = []
        results.append(("Простой запрос", test_simple()))
        results.append(("Названия глав", test_chapter_naming()))
        results.append(("Суммаризация", test_summarization()))

        # Итоги
        print("=" * 50)
        print("📊 Результаты тестов:")
        print("=" * 50)
        for name, result in results:
            status = "✅" if result else "⚠️"
            print(f"{status} {name}")

        successful = sum(1 for _, r in results if r)
        print(f"\nУспешно: {successful}/{len(results)}")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Убедитесь что:")
        print("   1. LM Studio запущен")
        print("   2. Сервер включён (Start Server)")
        print("   3. Загружена модель")
        print("   4. Сервер слушает на http://localhost:1234")
        sys.exit(1)

if __name__ == "__main__":
    main()
