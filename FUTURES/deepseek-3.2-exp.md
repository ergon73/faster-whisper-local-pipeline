Отличный проект! Проблема с плохо структурированным текстом после транскрибации очень распространена. Вот несколько решений для улучшения качества и структурирования выходных данных:

## 🎯 Проблемы текущей транскрибации и решения

### 1. **Сегментация по времени и говорящим**

```python
# Добавьте в конфигурацию
@dataclass(slots=True)
class EnhancedPipelineConfig(PipelineConfig):
    segment_duration: int = 30  # секунды
    speaker_diarization: bool = False
    min_silence_duration: int = 1000  # мс для разделения говорящих
    output_format: str = "structured"  # structured, srt, txt
```

### 2. **Постобработка текста**

Создайте модуль `text_postprocessor.py`:

```python
import re
from typing import List, Dict
import spacy
from collections import defaultdict

class TextPostProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("ru_core_news_sm")
        except OSError:
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Очистка и нормализация текста"""
        # Удаление повторяющихся фраз
        text = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', text)
        
        # Удаление междометий и слов-паразитов
        filler_words = {'ну', 'короче', 'типа', 'вот', 'это', 'как бы'}
        words = text.split()
        words = [w for w in words if w.lower() not in filler_words]
        
        return ' '.join(words)
    
    def split_into_paragraphs(self, text: str, max_length: int = 500) -> List[str]:
        """Разбивка на абзацы по смыслу"""
        if not self.nlp:
            # Простая разбивка по точкам
            sentences = re.split(r'[.!?]+', text)
            paragraphs = []
            current_para = []
            current_len = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if current_len + len(sentence) > max_length and current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
                    current_len = 0
                
                current_para.append(sentence)
                current_len += len(sentence)
            
            if current_para:
                paragraphs.append(' '.join(current_para))
            
            return paragraphs
        
        # Использование spacy для более интеллектуальной сегментации
        doc = self.nlp(text)
        paragraphs = []
        current_para = []
        current_len = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            if current_len + len(sent_text) > max_length and current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
                current_len = 0
            
            current_para.append(sent_text)
            current_len += len(sent_text)
        
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        return paragraphs
    
    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """Извлечение ключевых фраз"""
        words = re.findall(r'\b[а-яё]{4,}\b', text.lower())
        word_freq = defaultdict(int)
        
        for word in words:
            if len(word) > 3:  # Игнорируем короткие слова
                word_freq[word] += 1
        
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
```

### 3. **Улучшенная транскрибация с временными метками**

Модифицируйте функцию транскрибации:

```python
def enhanced_transcribe_audio(
    job: TranscriptionJob,
    output_path: Path,
    model: WhisperModel,
    config: EnhancedPipelineConfig,
    logger: logging.Logger,
) -> bool:
    """Улучшенная транскрибация с структурированием"""
    logger.info("Начало улучшенной транскрибации: %s", job.audio_path.name)
    start_time = time.perf_counter()

    try:
        segments, info = model.transcribe(
            str(job.audio_path),
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
            vad_parameters={"min_silence_duration_ms": config.vad_min_silence_ms},
        )

        post_processor = TextPostProcessor()
        
        # Собираем сегменты с временными метками
        structured_segments = []
        full_text = ""
        
        for segment in segments:
            text = segment.text.strip()
            if text:
                structured_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': text,
                    'cleaned_text': post_processor.clean_text(text)
                })
                full_text += text + " "

        # Постобработка
        paragraphs = post_processor.split_into_paragraphs(full_text)
        key_phrases = post_processor.extract_key_phrases(full_text)
        
        # Сохранение в структурированном формате
        output_data = {
            'metadata': {
                'source_file': job.source_path.name,
                'audio_file': job.audio_path.name,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'processing_time': time.perf_counter() - start_time
            },
            'key_phrases': [phrase for phrase, count in key_phrases],
            'structured_content': {
                'paragraphs': paragraphs,
                'segments': structured_segments
            }
        }
        
        # Сохранение в разных форматах
        if config.output_format == "json":
            with output_path.with_suffix('.json').open('w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        elif config.output_format == "structured":
            # Markdown-формат с временными метками
            with output_path.with_suffix('.md').open('w', encoding='utf-8') as f:
                f.write(f"# Транскрибация: {job.source_path.name}\n\n")
                f.write(f"**Длительность:** {info.duration:.2f} сек\n")
                f.write(f"**Язык:** {info.language} (вероятность: {info.language_probability:.2f})\n")
                f.write(f"**Ключевые темы:** {', '.join(output_data['key_phrases'][:5])}\n\n")
                
                f.write("## Содержание\n\n")
                for i, para in enumerate(paragraphs, 1):
                    f.write(f"{i}. {para[:100]}...\n")
                
                f.write("\n## Полный текст\n\n")
                for i, para in enumerate(paragraphs, 1):
                    f.write(f"### Абзац {i}\n\n{para}\n\n")
        
        elif config.output_format == "srt":
            # Субтитры
            with output_path.with_suffix('.srt').open('w', encoding='utf-8') as f:
                for i, segment in enumerate(structured_segments, 1):
                    start_time = format_timestamp(segment['start'])
                    end_time = format_timestamp(segment['end'])
                    f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n")
        
        else:  # txt по умолчанию
            with output_path.open('w', encoding='utf-8') as f:
                f.write(f"Транскрибация: {job.source_path.name}\n")
                f.write("=" * 50 + "\n\n")
                for para in paragraphs:
                    f.write(para + "\n\n")

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Улучшенная транскрибация завершена: %s (%.2f сек)",
            job.audio_path.name,
            elapsed,
        )
        return True
        
    except Exception as exc:
        logger.exception("Ошибка улучшенной транскрибации %s", job.audio_path.name)
        return False

def format_timestamp(seconds: float) -> str:
    """Форматирование времени для SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
```

### 4. **Интеграция с векторными базами данных**

Для долгосрочного хранения и поиска:

```python
# requirements_additional.txt
# chromadb>=0.4.0
# sentence-transformers>=2.2.0

class KnowledgeBase:
    def __init__(self, persist_directory: str = "./knowledge_base"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
    def store_transcription(self, transcription_data: dict, source_info: dict):
        """Сохраняет транскрибацию в векторной БД"""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            client = chromadb.PersistentClient(path=str(self.persist_directory))
            collection = client.get_or_create_collection("webinar_transcriptions")
            
            # Генерация эмбеддингов для поиска
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            embeddings = model.encode(transcription_data['full_text'])
            
            collection.add(
                embeddings=[embeddings.tolist()],
                documents=[transcription_data['full_text']],
                metadatas=[{
                    'source_file': source_info['source_file'],
                    'duration': source_info['duration'],
                    'language': source_info['language'],
                    'timestamp': time.time(),
                    'key_phrases': transcription_data['key_phrases']
                }],
                ids=[source_info['source_file']]
            )
            
        except ImportError:
            print("Для работы векторной БД установите chromadb и sentence-transformers")
```

### 5. **Обновленный основной скрипт**

```python
def process_enhanced_files(config: EnhancedPipelineConfig, logger: logging.Logger) -> None:
    """Улучшенный пайплайн с постобработкой"""
    logger.info("Запуск улучшенного пайплайна Faster-Whisper Local")
    ensure_directories(config)
    
    if media_requires_ffmpeg(config):
        try:
            ensure_ffmpeg_available(logger)
        except RuntimeError as exc:
            logger.error("Нужен FFmpeg для видео/сжатых аудио: %s", exc)
            return

    jobs = prepare_jobs(config, logger)
    if not jobs:
        logger.warning("Не найдено файлов для обработки")
        return

    logger.info("Найдено задач для транскрибации: %d", len(jobs))

    try:
        model = load_model(config, logger)
    except Exception as exc:
        logger.error("Не удалось загрузить модель Whisper: %s", exc)
        return

    # Инициализация базы знаний
    knowledge_base = KnowledgeBase()
    
    success = 0
    for index, job in enumerate(jobs, start=1):
        logger.info("[%d/%d] Обработка %s", index, len(jobs), job.audio_path.name)
        
        result_path = config.transcripts_out / f"{job.audio_path.stem}.{config.output_format}"
        if result_path.exists() and result_path.stat().st_mtime >= job.audio_path.stat().st_mtime:
            logger.info("Пропуск транскрибации (актуальный результат уже существует): %s", result_path.name)
            success += 1
            continue
            
        if enhanced_transcribe_audio(job, result_path, model, config, logger):
            success += 1
            logger.info("Результат сохранен: %s", result_path.name)
            
            # Сохранение в базу знаний (опционально)
            if config.enable_knowledge_base:
                try:
                    # Загрузка и сохранение в векторную БД
                    with result_path.with_suffix('.json').open('r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    knowledge_base.store_transcription(
                        data['structured_content'],
                        data['metadata']
                    )
                    logger.info("Данные сохранены в базу знаний")
                except Exception as e:
                    logger.warning("Не удалось сохранить в базу знаний: %s", e)
        else:
            logger.error("Ошибка при обработке %s", job.audio_path.name)

    logger.info("Обработка завершена: %d/%d файлов успешно", success, len(jobs))
```

## 🚀 Быстрый старт улучшений

1. **Установите дополнительные зависимости:**
```bash
pip install spacy sentence-transformers chromadb
python -m spacy download ru_core_news_sm
```

2. **Обновите конфигурацию в `.env`:**
```env
# Новые параметры
OUTPUT_FORMAT=structured  # structured, srt, json, txt
ENABLE_KNOWLEDGE_BASE=true
SEGMENT_DURATION=30
MIN_SILENCE_DURATION=1000
```

3. **Запустите улучшенную версию:**
```python
# enhanced_transcribe.py
from text_postprocessor import TextPostProcessor
from knowledge_base import KnowledgeBase

# Используйте enhanced_transcribe_audio вместо стандартной
```

## 📊 Результат после улучшений

Вместо сплошного текста вы получите:

### Структурированный Markdown:
```markdown
# Транскрибация: webinar_2024.mp4

**Длительность:** 7200 сек
**Язык:** ru (вероятность: 0.98)
**Ключевые темы:** python, gpt, nlp, pytorch, обучение

## Содержание

1. Введение и приветствие участников...
2. Обзор тем предстоящих вебинаров...
3. Демонстрация работы с Python и GPT...

## Полный текст

### Абзац 1
Всем привет, всем привет. Давайте плюсы, вопросы, ответы...

### Абзац 2
Сегодня как кодить на Python с помощью GPT и будут прям прикольные штуки...
```

### JSON с метаданными:
```json
{
  "metadata": {
    "source_file": "webinar_2024.mp4",
    "duration": 7200.5,
    "language": "ru",
    "processing_time": 45.23
  },
  "key_phrases": ["python", "gpt", "nlp", "pytorch"],
  "structured_content": {
    "paragraphs": ["Текст абзаца 1...", "Текст абзаца 2..."],
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "text": "Всем привет, всем привет...",
        "cleaned_text": "Всем привет..."
      }
    ]
  }
}
```

## 💡 Преимущества этого подхода

1. **Структурирование** - разбивка на логические абзацы
2. **Очистка текста** - удаление повторов и слов-паразитов
3. **Извлечение ключевых тем** - быстрый поиск по содержанию
4. **Временные метки** - навигация по оригинальному видео
5. **Векторный поиск** - семантический поиск по всем вебинарам
6. **Мультиформатность** - разные форматы для разных целей

Ваши 300+ вебинаров теперь превратятся в структурированную базу знаний, где вы сможете быстро находить нужную информацию! 🎉