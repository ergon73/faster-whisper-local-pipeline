# Решение: Структурирование и индексация транскриптов

Ваша проблема типична для raw транскриптов! Вот комплексное решение для превращения хаотичного текста в полезную knowledge base.

## 🎯 Предлагаемая архитектура

```
Raw Transcript → AI Processing → Structured Output → Vector DB → Search/RAG
```

---

## 📦 Решение 1: Post-Processing Pipeline (рекомендую начать с этого)

Добавьте в проект модуль для постобработки транскриптов с помощью LLM.

### Новый файл: `postprocess.py`

```python
"""
Постобработка транскриптов: очистка, структурирование, извлечение метаданных.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class ProcessedTranscript:
    """Структурированный результат обработки транскрипта."""
    
    title: str
    summary: str
    key_topics: list[str]
    sections: list[dict[str, str]]  # [{"title": "...", "content": "...", "timestamp": "..."}]
    key_quotes: list[str]
    action_items: list[str]
    metadata: dict[str, str | list[str]]


class TranscriptProcessor:
    """Обработчик транскриптов с использованием LLM."""
    
    def __init__(
        self,
        provider: str = "anthropic",  # "anthropic", "openai", или "local"
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.provider = provider
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        
        if provider == "anthropic":
            if anthropic is None:
                raise ImportError("Установите: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            if OpenAI is None:
                raise ImportError("Установите: pip install openai")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None  # для локальных моделей
    
    def _create_processing_prompt(self, raw_text: str) -> str:
        """Создаёт промпт для структурирования транскрипта."""
        return f"""Ты получил RAW транскрипт учебного вебинара/лекции. Твоя задача - превратить его в хорошо структурированный учебный материал.

ИСХОДНЫЙ ТРАНСКРИПТ:
{raw_text[:30000]}  # ограничение для контекста

ЗАДАЧИ:
1. Удали verbal fillers (э-э, м-м, ну, типа, короче) и повторы
2. Разбей на логические разделы с заголовками
3. Выдели ключевые темы (keywords)
4. Создай краткое саммари (3-5 предложений)
5. Извлеки важные цитаты и факты
6. Найди action items / рекомендации (если есть)

ФОРМАТ ОТВЕТА (строго JSON):
{{
  "title": "Название вебинара",
  "summary": "Краткое описание содержания",
  "key_topics": ["тема1", "тема2", ...],
  "sections": [
    {{"title": "Раздел 1", "content": "Очищенный текст раздела", "timestamp": "примерное время"}},
    ...
  ],
  "key_quotes": ["цитата1", "цитата2", ...],
  "action_items": ["рекомендация1", ...],
  "metadata": {{
    "difficulty": "beginner|intermediate|advanced",
    "duration": "примерная длительность",
    "technologies": ["Python", "PyTorch", ...]
  }}
}}

Отвечай ТОЛЬКО валидным JSON, без комментариев до и после."""

    def _call_llm(self, prompt: str) -> str:
        """Вызывает LLM API."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        
        else:
            raise NotImplementedError(f"Provider '{self.provider}' not implemented")
    
    def process_transcript(self, raw_text: str) -> ProcessedTranscript:
        """Обрабатывает один транскрипт."""
        self.logger.info("Начало обработки транскрипта (длина: %d символов)", len(raw_text))
        
        prompt = self._create_processing_prompt(raw_text)
        
        try:
            response = self._call_llm(prompt)
            
            # Извлекаем JSON из ответа (на случай если модель добавила текст)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            data = json.loads(response)
            
            return ProcessedTranscript(
                title=data.get("title", "Без названия"),
                summary=data.get("summary", ""),
                key_topics=data.get("key_topics", []),
                sections=data.get("sections", []),
                key_quotes=data.get("key_quotes", []),
                action_items=data.get("action_items", []),
                metadata=data.get("metadata", {}),
            )
        
        except json.JSONDecodeError as exc:
            self.logger.error("Не удалось разобрать JSON от LLM: %s", exc)
            self.logger.debug("Ответ LLM: %s", response[:500])
            raise
        except Exception as exc:
            self.logger.exception("Ошибка обработки: %s", exc)
            raise
    
    def save_processed(
        self,
        processed: ProcessedTranscript,
        output_dir: Path,
        filename_stem: str,
    ) -> None:
        """Сохраняет обработанный транскрипт в нескольких форматах."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON (машиночитаемый)
        json_path = output_dir / f"{filename_stem}_processed.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(processed), f, ensure_ascii=False, indent=2)
        self.logger.info("Сохранён JSON: %s", json_path)
        
        # Markdown (человекочитаемый)
        md_path = output_dir / f"{filename_stem}_structured.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# {processed.title}\n\n")
            f.write(f"## 📝 Краткое содержание\n\n{processed.summary}\n\n")
            
            if processed.key_topics:
                f.write(f"## 🔑 Ключевые темы\n\n")
                for topic in processed.key_topics:
                    f.write(f"- {topic}\n")
                f.write("\n")
            
            f.write(f"## 📚 Содержание\n\n")
            for i, section in enumerate(processed.sections, 1):
                title = section.get("title", f"Раздел {i}")
                content = section.get("content", "")
                timestamp = section.get("timestamp", "")
                
                f.write(f"### {title}")
                if timestamp:
                    f.write(f" `[{timestamp}]`")
                f.write("\n\n")
                f.write(f"{content}\n\n")
            
            if processed.key_quotes:
                f.write(f"## 💡 Важные цитаты\n\n")
                for quote in processed.key_quotes:
                    f.write(f"> {quote}\n\n")
            
            if processed.action_items:
                f.write(f"## ✅ Action Items\n\n")
                for item in processed.action_items:
                    f.write(f"- [ ] {item}\n")
        
        self.logger.info("Сохранён Markdown: %s", md_path)


def process_all_transcripts(
    input_dir: Path,
    output_dir: Path,
    processor: TranscriptProcessor,
    logger: logging.Logger,
) -> None:
    """Обрабатывает все транскрипты в директории."""
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        logger.warning("Не найдено TXT файлов в %s", input_dir)
        return
    
    logger.info("Найдено файлов для обработки: %d", len(txt_files))
    
    for i, txt_file in enumerate(txt_files, 1):
        logger.info("[%d/%d] Обработка: %s", i, len(txt_files), txt_file.name)
        
        try:
            raw_text = txt_file.read_text(encoding="utf-8")
            processed = processor.process_transcript(raw_text)
            processor.save_processed(processed, output_dir, txt_file.stem)
            logger.info("✓ Успешно обработан: %s", txt_file.name)
        
        except Exception as exc:
            logger.error("✗ Ошибка обработки %s: %s", txt_file.name, exc)
            continue


def main() -> None:
    """CLI для постобработки."""
    import os
    from transcribe_v2 import configure_logging
    
    logger = configure_logging()
    
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Установите ANTHROPIC_API_KEY или OPENAI_API_KEY")
        return
    
    provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
    model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022" if provider == "anthropic" else "gpt-4o")
    
    processor = TranscriptProcessor(
        provider=provider,
        model=model,
        api_key=api_key,
        logger=logger,
    )
    
    input_dir = Path(os.getenv("TRANSCRIBE_OUT", "transcribe"))
    output_dir = Path(os.getenv("PROCESSED_OUT", "processed"))
    
    process_all_transcripts(input_dir, output_dir, processor, logger)
    logger.info("Обработка завершена. Результаты в %s", output_dir)


if __name__ == "__main__":
    main()
```

### Обновите `.env`:

```bash
# Добавьте в .env
ANTHROPIC_API_KEY=your_key_here
# или
OPENAI_API_KEY=your_key_here

LLM_MODEL=claude-3-5-sonnet-20241022
PROCESSED_OUT=processed
```

### Установите зависимости:

```bash
pip install anthropic  # или openai
```

### Использование:

```bash
# Сначала транскрибация
python transcribe_v2.py

# Затем постобработка
python postprocess.py
```

---

## 📦 Решение 2: RAG Knowledge Base (для поиска по 300 вебинарам)

### Новый файл: `knowledge_base.py`

```python
"""
Векторная база знаний для поиска по транскриптам.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None


class TranscriptKnowledgeBase:
    """Векторная база для поиска по транскриптам."""
    
    def __init__(
        self,
        db_path: Path = Path("chroma_db"),
        collection_name: str = "webinars",
        logger: Optional[logging.Logger] = None,
    ):
        if chromadb is None:
            raise ImportError("Установите: pip install chromadb")
        
        self.logger = logger or logging.getLogger(__name__)
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        # Используем sentence-transformers для эмбеддингов
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-large"  # хорош для русского
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"description": "Webinar transcripts knowledge base"}
        )
        
        self.logger.info("Инициализирована база: %s (документов: %d)", 
                        collection_name, self.collection.count())
    
    def index_processed_transcript(self, json_path: Path) -> None:
        """Индексирует обработанный транскрипт из JSON."""
        data = json.loads(json_path.read_text(encoding="utf-8"))
        
        doc_id = json_path.stem
        
        # Индексируем разделы отдельно для точного поиска
        for i, section in enumerate(data.get("sections", [])):
            section_id = f"{doc_id}_section_{i}"
            
            metadata = {
                "source": json_path.name,
                "title": data.get("title", ""),
                "section_title": section.get("title", ""),
                "timestamp": section.get("timestamp", ""),
                "topics": ",".join(data.get("key_topics", [])),
            }
            
            content = f"{section.get('title', '')}\n\n{section.get('content', '')}"
            
            self.collection.upsert(
                ids=[section_id],
                documents=[content],
                metadatas=[metadata],
            )
        
        self.logger.info("Проиндексирован: %s (%d разделов)", 
                        data.get("title"), len(data.get("sections", [])))
    
    def index_directory(self, processed_dir: Path) -> None:
        """Индексирует все JSON файлы в директории."""
        json_files = list(processed_dir.glob("*_processed.json"))
        
        self.logger.info("Найдено файлов для индексации: %d", len(json_files))
        
        for json_file in json_files:
            try:
                self.index_processed_transcript(json_file)
            except Exception as exc:
                self.logger.error("Ошибка индексации %s: %s", json_file.name, exc)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_topics: Optional[list[str]] = None,
    ) -> list[dict]:
        """Ищет релевантные фрагменты."""
        where = None
        if filter_topics:
            # Простой фильтр (можно улучшить)
            where = {"topics": {"$contains": filter_topics[0]}}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        
        return formatted


def main() -> None:
    """CLI для индексации и поиска."""
    import os
    from transcribe_v2 import configure_logging
    
    logger = configure_logging()
    
    kb = TranscriptKnowledgeBase(logger=logger)
    
    processed_dir = Path(os.getenv("PROCESSED_OUT", "processed"))
    
    if processed_dir.exists():
        kb.index_directory(processed_dir)
    
    # Интерактивный поиск
    print("\n🔍 Поиск по базе знаний (Ctrl+C для выхода)\n")
    
    while True:
        try:
            query = input("Ваш вопрос: ").strip()
            if not query:
                continue
            
            results = kb.search(query, n_results=3)
            
            print(f"\n📚 Найдено результатов: {len(results)}\n")
            
            for i, result in enumerate(results, 1):
                print(f"[{i}] {result['metadata'].get('title', 'N/A')}")
                print(f"    Раздел: {result['metadata'].get('section_title', 'N/A')}")
                print(f"    Релевантность: {1 - result['distance']:.2%}")
                print(f"\n{result['content'][:300]}...\n")
                print("-" * 80)
        
        except KeyboardInterrupt:
            print("\n\nДо встречи!")
            break


if __name__ == "__main__":
    main()
```

### Установка:

```bash
pip install chromadb sentence-transformers
```

### Использование:

```bash
# Индексация всех обработанных транскриптов
python knowledge_base.py

# Затем можно искать интерактивно
```

---

## 🚀 Полный Pipeline

### Обновите `requirements.txt`:

```
faster-whisper>=1.0.0
python-dotenv
pytest
anthropic  # или openai
chromadb
sentence-transformers
```

### Создайте `pipeline.py` (all-in-one):

```python
"""
Полный пайплайн: транскрибация → структурирование → индексация.
"""

import os
from pathlib import Path
from transcribe_v2 import PipelineConfig, process_files, configure_logging
from postprocess import TranscriptProcessor, process_all_transcripts
from knowledge_base import TranscriptKnowledgeBase


def run_full_pipeline() -> None:
    """Запускает весь pipeline обработки."""
    logger = configure_logging()
    
    logger.info("=" * 60)
    logger.info("ЭТАП 1: Транскрибация видео/аудио")
    logger.info("=" * 60)
    
    config = PipelineConfig.from_env()
    process_files(config, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("ЭТАП 2: Структурирование транскриптов")
    logger.info("=" * 60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
        model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        
        processor = TranscriptProcessor(
            provider=provider,
            model=model,
            api_key=api_key,
            logger=logger,
        )
        
        input_dir = Path(os.getenv("TRANSCRIBE_OUT", "transcribe"))
        output_dir = Path(os.getenv("PROCESSED_OUT", "processed"))
        
        process_all_transcripts(input_dir, output_dir, processor, logger)
    else:
        logger.warning("Пропуск структурирования (нет API ключа). Установите ANTHROPIC_API_KEY или OPENAI_API_KEY")
    
    logger.info("\n" + "=" * 60)
    logger.info("ЭТАП 3: Индексация в векторную БД")
    logger.info("=" * 60)
    
    try:
        kb = TranscriptKnowledgeBase(logger=logger)
        processed_dir = Path(os.getenv("PROCESSED_OUT", "processed"))
        
        if processed_dir.exists():
            kb.index_directory(processed_dir)
            logger.info("✓ Индексация завершена. Используйте knowledge_base.py для поиска")
        else:
            logger.warning("Директория %s не найдена", processed_dir)
    
    except ImportError:
        logger.warning("Пропуск индексации (chromadb не установлен)")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE ЗАВЕРШЁН")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()
```

### Запуск всего pipeline:

```bash
python pipeline.py
```

---

## 📊 Результат

После обработки вашего образца получится:

### `meeting_video_mp4_structured.md`:

```markdown
# Вебинар: Как кодить на Python с помощью GPT

## 📝 Краткое содержание

Вебинар посвящён практическому применению GPT для программирования на Python. Рассматриваются инструменты автоматизации кода, классические подходы NLP, обучение с подкреплением и интересные новости из области нейрокомпьютерных интерфейсов.

## 🔑 Ключевые темы

- GPT для программирования
- Классический NLP (BERT, T5)
- PyTorch
- Обучение с подкреплением
- Нейрокомпьютерные интерфейсы

## 📚 Содержание

### Введение и расписание `[00:00]`

Приветствие участников и анонс программы на следующие две недели. Планируется четыре вебинара по темам: программирование с GPT, классический NLP, PyTorch и обучение с подкреплением.

### Промо-акция и новости `[05:30]`

Анонс новой рулетки в Telegram-боте для участников курса...

### Новость: нейрокомпьютерные интерфейсы `[10:15]`

Рассказ о компании NIR, которая работает с энцефалограммами и нейроинтерфейсами. Описан эксперимент с мышью, которая научилась отвечать на вопросы через прямую стимуляцию мозга. LLM генерирует ответы, которые преобразуются в нейронные сигналы...

## 💡 Важные цитаты

> "Обучение с подкреплением — это когда нейронка обучается в реальном среде в реальном времени"

> "Дофаминовая система — это мощная система обучения с подкреплением в реальной жизни"

## ✅ Action Items

- [ ] Изучить PyTorch для создания продвинутых нейронных сетей
- [ ] Рассмотреть применение классических NLP моделей (BERT, T5) для своих задач
```

---

## 💰 Стоимость обработки

**Claude 3.5 Sonnet:**
- Вход: $3 за 1M токенов
- Выход: $15 за 1M токенов
- Один вебинар (~30K токенов): ~$0.10-0.20

**GPT-4o:**
- Вход: $2.50 за 1M токенов  
- Выход: $10 за 1M токенов
- Один вебинар: ~$0.08-0.15

**Для 300 вебинаров: $24-60** (единоразово)

---

## 🎁 Бонус: Локальная альтернатива (бесплатно, но медленнее)

Используйте **Ollama** с `llama3.2` или `mistral`:

```python
def _call_local_llm(self, prompt: str) -> str:
    """Вызов локальной модели через Ollama."""
    import requests
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
        }
    )
    return response.json()["response"]
```

---

## 📈 Что в итоге?

1. ✅ **Структурированные материалы** в Markdown
2. ✅ **Машиночитаемые данные** в JSON
3. ✅ **Поиск по 300 вебинарам** за секунды
4. ✅ **Экспорт в Notion/Obsidian** (Markdown)
5. ✅ **RAG-чатбот** для вопросов по курсу

Запускайте `python pipeline.py` и получайте полезную knowledge base из всех вебинаров! 🚀