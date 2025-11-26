"""Анализ проблемных глав День 3."""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Загрузим главы Дня 3
with open('transcribe/Большой марафон по Классическому AI. День 3_video_mp4_chapters.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Загрузим абзацы
paragraphs = {}
with open('transcribe/Большой марафон по Классическому AI. День 3_video_mp4_packed.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line.strip())
        if row.get('type') == 'paragraph':
            paragraphs[row['id']] = row['text']

print('=== АНАЛИЗ ПРОБЛЕМНЫХ ГЛАВ ДЕНЬ 3 (8-12) ===\n')

for idx in range(7, 12):  # Главы 8-12 (индексы 7-11)
    chapter = data['chapters'][idx]
    full_text = '\n\n'.join(paragraphs[pid] for pid in chapter['paragraph_ids'] if pid in paragraphs)

    print(f'\n{"="*80}')
    print(f'ГЛАВА {idx+1}: {chapter["title"]}')
    print(f'{"="*80}')
    print(f'Длина: {len(full_text)} символов')
    print(f'Абзацев: {len(chapter["paragraph_ids"])}')

    # Проверим на специфические паттерны
    first_500 = full_text[:500]

    # 1. Признаки live coding / демонстрации
    demo_words = ['смотрим', 'смотрите', 'вот так', 'сейчас', 'давайте', 'запускаем']
    has_demo = sum(1 for word in demo_words if word in first_500.lower())

    # 2. Упоминания кода
    code_words = ['код', 'импорт', 'класс', 'функция', 'def ', 'import']
    has_code_refs = sum(1 for word in code_words if word in first_500.lower())

    # 3. Проверим на технические термины PyTorch
    pytorch_terms = ['пайторч', 'pytorch', 'тензор', 'tensor', 'слой', 'layer', 'нейронк']
    pytorch_count = sum(1 for term in pytorch_terms if term in first_500.lower())

    # 4. Проверим качество транскрипции (ошибки STT)
    stt_errors = ['эээ', 'ммм', 'эм', 'типа того', 'ну вот']
    stt_error_count = sum(1 for err in stt_errors if err in first_500.lower())

    print(f'\nАНАЛИЗ ПЕРВЫХ 500 СИМВОЛОВ:')
    print(f'  Признаки демонстрации: {has_demo} упоминаний')
    print(f'  Упоминания кода: {has_code_refs} упоминаний')
    print(f'  PyTorch термины: {pytorch_count} упоминаний')
    print(f'  Ошибки STT: {stt_error_count} упоминаний')

    print(f'\nПЕРВЫЕ 300 СИМВОЛОВ:')
    print(full_text[:300])

    print(f'\nПОСЛЕДНИЕ 200 СИМВОЛОВ:')
    print(full_text[-200:])

# Теперь сравним с нормальными главами
print(f'\n\n{"#"*80}')
print('СРАВНЕНИЕ С НОРМАЛЬНЫМИ ГЛАВАМИ')
print(f'{"#"*80}\n')

# Возьмем главы 6, 7 (до проблемных) и 13, 14 (после проблемных)
normal_indices = [5, 6, 12, 13]

for idx in normal_indices:
    if idx < len(data['chapters']):
        chapter = data['chapters'][idx]
        full_text = '\n\n'.join(paragraphs[pid] for pid in chapter['paragraph_ids'] if pid in paragraphs)

        print(f'\nГлава {idx+1}: {chapter["title"][:50]}...')
        print(f'  Длина: {len(full_text)} символов')

        first_500 = full_text[:500]
        demo_words = ['смотрим', 'смотрите', 'вот так', 'сейчас', 'давайте', 'запускаем']
        has_demo = sum(1 for word in demo_words if word in first_500.lower())

        print(f'  Признаки демонстрации: {has_demo}')
        print(f'  Первые 150 символов: {full_text[:150]}')
