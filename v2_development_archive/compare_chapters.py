"""Сравнение успешной и проблемной глав."""
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

# Глава 1 (успешная) vs Глава 8 (проблемная)
chapters_to_compare = [
    (0, "УСПЕШНАЯ"),
    (7, "ПРОБЛЕМНАЯ")
]

for idx, status in chapters_to_compare:
    chapter = data['chapters'][idx]
    full_text = '\n\n'.join(paragraphs[pid] for pid in chapter['paragraph_ids'] if pid in paragraphs)

    print(f'\n{"="*80}')
    print(f'{status}: ГЛАВА {idx+1} - {chapter["title"]}')
    print(f'{"="*80}')

    print(f'\nСТАТИСТИКА:')
    print(f'  Длина: {len(full_text)} символов')
    print(f'  Абзацев: {len(chapter["paragraph_ids"])}')
    print(f'  Средняя длина абзаца: {len(full_text) // len(chapter["paragraph_ids"])} символов')

    # Анализ плотности технических терминов
    words = full_text.split()
    print(f'  Всего слов: {len(words)}')
    print(f'  Средняя длина слова: {sum(len(w) for w in words) / len(words):.2f}')

    # Подсчет переносов строк (структурированность)
    paragraphs_count = full_text.count('\n\n')
    print(f'  Абзацев (по \\n\\n): {paragraphs_count}')

    # Проверим на очень длинные абзацы (монолитность)
    text_parts = full_text.split('\n\n')
    long_parts = [p for p in text_parts if len(p) > 1000]
    print(f'  Очень длинных блоков (>1000 символов): {len(long_parts)}')

    # Анализ начала (первые 1000 символов)
    first_1000 = full_text[:1000]

    # Подсчет технических терминов
    tech_terms = ['pytorch', 'пайторч', 'тензор', 'tensor', 'слой', 'layer', 'градиент',
                  'optimizer', 'backward', 'forward', 'batch', 'батч', 'эпох', 'epoch']
    tech_count = sum(first_1000.lower().count(term) for term in tech_terms)
    print(f'  Техн. термины в первых 1000 символов: {tech_count}')

    # Подсчет "воды" (пояснительных фраз)
    filler_phrases = ['то есть', 'например', 'другими словами', 'проще говоря',
                      'представьте', 'давайте', 'можно', 'это как']
    filler_count = sum(first_1000.lower().count(phrase) for phrase in filler_phrases)
    print(f'  Пояснительные фразы в первых 1000 символов: {filler_count}')

    # Соотношение техника / вода
    if filler_count > 0:
        ratio = tech_count / filler_count
        print(f'  Соотношение техника/вода: {ratio:.2f}')
    else:
        print(f'  Соотношение техника/вода: бесконечность (нет пояснений!)')

    print(f'\nПЕРВЫЕ 800 СИМВОЛОВ:')
    print(full_text[:800])
    print('\n[...]')
