
## Анализ проблемы и решение

Вы столкнулись с типичной проблемой при работе с транскрибациями длинных аудиозаписей - неструктурированный текст, который трудно читать и использовать. Но не волнуйтесь, ваши 300+ вебинаров не пропали! Я предложу вам комплексное решение, которое превратит их в ценную базу знаний.

### Основная проблема

Транскрипции от Whisper создают сплошной текст без:
- Структурирования по темам
- Выделения ключевых моментов
- Форматирования для удобного чтения
- Возможности быстрого поиска информации

### Комплексное решение

Я предлагаю разработать дополнительный модуль для пост-обработки транскрибаций, который будет:

1. **Структурировать текст** по темам и смысловым блокам
2. **Создавать оглавление** для навигации
3. **Выделять ключевые тезисы** и основные моменты
4. **Формировать краткие резюме** каждого вебинара
5. **Индексировать содержимое** для быстрого поиска
6. **Создавать интерактивную базу знаний** с возможностью поиска по семантике

### План реализации

Давайте создадим новый скрипт `process_transcripts.py`, который будет обрабатывать файлы транскрипций и преобразовывать их в структурированные документы.

Вот пример реализации:

```python
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Установка необходимых NLTK пакетов при первом запуске
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class TranscriptProcessor:
    """Класс для обработки транскрипций вебинаров"""
    
    input_dir: Path = Path("transcribe")
    output_dir: Path = Path("processed_transcripts")
    model_name: str = "all-MiniLM-L6-v2"  # Модель для создания эмбеддингов
    min_cluster_size: int = 5  # Минимальный размер кластера для создания темы
    max_topics: int = 10  # Максимальное количество тем
    language: str = "russian"
    
    def __post_init__(self):
        """Инициализация после создания объекта"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(self.model_name)
        self.stop_words = set(stopwords.words(self.language))
    
    def process_all_transcripts(self) -> None:
        """Обработка всех файлов транскрипций в директории"""
        for file_path in self.input_dir.glob("*.txt"):
            self.process_single_transcript(file_path)
    
    def process_single_transcript(self, file_path: Path) -> None:
        """Обработка одного файла транскрипции"""
        logging.info(f"Обработка файла: {file_path.name}")
        
        # Чтение и предварительная обработка текста
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Разделение на предложения
        sentences = sent_tokenize(raw_text)
        
        # Создание эмбеддингов для предложений
        embeddings = self.model.encode(sentences)
        
        # Кластеризация предложений по темам
        topics = self._cluster_sentences(sentences, embeddings)
        
        # Определение ключевых слов для каждой темы
        topic_keywords = self._extract_keywords(sentences, topics)
        
        # Создание структурированного документа
        structured_doc = self._create_structured_document(
            file_path.stem, sentences, topics, topic_keywords
        )
        
        # Сохранение результатов
        self._save_results(structured_doc, file_path.stem)
        
        logging.info(f"Обработка завершена для: {file_path.name}")
    
    def _cluster_sentences(self, sentences: List[str], embeddings: np.ndarray) -> List[int]:
        """Кластеризация предложений по темам"""
        # Определение оптимального количества кластеров
        n_clusters = min(self.max_topics, max(2, len(sentences) // self.min_cluster_size))
        
        # Кластеризация с помощью K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        topics = kmeans.fit_predict(embeddings)
        
        return topics.tolist()
    
    def _extract_keywords(self, sentences: List[str], topics: List[int]) -> Dict[int, List[str]]:
        """Извлечение ключевых слов для каждой темы"""
        topic_sentences = {}
        for sentence, topic in zip(sentences, topics):
            if topic not in topic_sentences:
                topic_sentences[topic] = []
            topic_sentences[topic].append(sentence)
        
        topic_keywords = {}
        for topic, sent_list in topic_sentences.items():
            # Объединение всех предложений темы в один текст
            topic_text = " ".join(sent_list)
            
            # Извлечение ключевых слов с помощью TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=10,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2)
            )
            try:
                tfidf_matrix = vectorizer.fit_transform([topic_text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                # Сортировка по TF-IDF скору
                top_indices = np.argsort(tfidf_scores)[::-1][:5]
                keywords = [feature_names[i] for i in top_indices]
                topic_keywords[topic] = keywords
            except ValueError:
                # Если текст слишком короткий или содержит только стоп-слова
                topic_keywords[topic] = []
        
        return topic_keywords
    
    def _create_structured_document(
        self, 
        title: str, 
        sentences: List[str], 
        topics: List[int], 
        topic_keywords: Dict[int, List[str]]
    ) -> Dict:
        """Создание структурированного документа"""
        # Группировка предложений по темам
        topic_sentences = {}
        for sentence, topic in zip(sentences, topics):
            if topic not in topic_sentences:
                topic_sentences[topic] = []
            topic_sentences[topic].append(sentence)
        
        # Создание оглавления
        toc = []
        for topic, sent_list in topic_sentences.items():
            keywords = topic_keywords.get(topic, [])
            topic_title = f"Тема {topic + 1}: {', '.join(keywords[:3])}"
            toc.append({"topic_id": topic, "title": topic_title})
        
        # Создание основного содержимого
        content = []
        for topic, sent_list in topic_sentences.items():
            keywords = topic_keywords.get(topic, [])
            topic_title = f"Тема {topic + 1}: {', '.join(keywords[:3])}"
            
            # Форматирование текста темы
            formatted_text = "\n".join(sent_list)
            
            content.append({
                "topic_id": topic,
                "title": topic_title,
                "keywords": keywords,
                "text": formatted_text
            })
        
        # Создание резюме
        summary = self._create_summary(sentences, topics, topic_keywords)
        
        return {
            "title": title,
            "summary": summary,
            "table_of_contents": toc,
            "content": content,
            "total_topics": len(topic_sentences),
            "total_sentences": len(sentences)
        }
    
    def _create_summary(
        self, 
        sentences: List[str], 
        topics: List[int], 
        topic_keywords: Dict[int, List[str]]
    ) -> str:
        """Создание краткого резюме вебинара"""
        # Подсчет количества предложений для каждой темы
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Сортировка тем по количеству предложений
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Формирование резюме на основе ключевых слов самых больших тем
        summary_parts = []
        for topic, count in sorted_topics[:3]:
            keywords = topic_keywords.get(topic, [])
            if keywords:
                summary_parts.append(f"{', '.join(keywords[:3])}")
        
        return f"Вебинар посвящен следующим темам: {', '.join(summary_parts)}."
    
    def _save_results(self, structured_doc: Dict, filename: str) -> None:
        """Сохранение результатов в различных форматах"""
        # Создание директории для результатов, если она не существует
        output_subdir = self.output_dir / filename
        output_subdir.mkdir(exist_ok=True)
        
        # Сохранение в JSON
        with open(output_subdir / f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(structured_doc, f, ensure_ascii=False, indent=2)
        
        # Сохранение в Markdown
        self._save_as_markdown(structured_doc, output_subdir / f"{filename}.md")
        
        # Сохранение в HTML
        self._save_as_html(structured_doc, output_subdir / f"{filename}.html")
        
        # Создание облака слов
        self._create_wordcloud(structured_doc, output_subdir / f"{filename}_wordcloud.png")
        
        # Создание визуализации тем
        self._create_topic_visualization(structured_doc, output_subdir / f"{filename}_topics.png")
    
    def _save_as_markdown(self, doc: Dict, output_path: Path) -> None:
        """Сохранение документа в формате Markdown"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {doc['title']}\n\n")
            
            # Добавление резюме
            f.write("## Резюме\n\n")
            f.write(f"{doc['summary']}\n\n")
            
            # Добавление оглавления
            f.write("## Содержание\n\n")
            for item in doc['table_of_contents']:
                f.write(f"- [{item['title']}](#topic-{item['topic_id'] + 1})\n")
            f.write("\n")
            
            # Добавление основного содержимого
            for item in doc['content']:
                topic_id = item['topic_id'] + 1
                f.write(f"## Тема {topic_id}: {item['title']}\n\n")
                
                if item['keywords']:
                    f.write("**Ключевые слова:** ")
                    f.write(", ".join(item['keywords']))
                    f.write("\n\n")
                
                f.write(f"{item['text']}\n\n")
    
    def _save_as_html(self, doc: Dict, output_path: Path) -> None:
        """Сохранение документа в формате HTML"""
        html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{doc['title']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .toc {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            margin: 5px 0;
        }}
        .toc a {{
            color: #3498db;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .topic {{
            margin-bottom: 30px;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
        }}
        .keywords {{
            background-color: #e8f4f8;
            border-radius: 3px;
            padding: 5px 10px;
            margin: 10px 0;
            display: inline-block;
        }}
        .summary {{
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 10px 15px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>{doc['title']}</h1>
    
    <div class="summary">
        <h2>Резюме</h2>
        <p>{doc['summary']}</p>
    </div>
    
    <div class="toc">
        <h2>Содержание</h2>
        <ul>
"""
        
        # Добавление оглавления
        for item in doc['table_of_contents']:
            html_content += f'            <li><a href="#topic-{item["topic_id"] + 1}">{item["title"]}</a></li>\n'
        
        html_content += """        </ul>
    </div>
"""
        
        # Добавление основного содержимого
        for item in doc['content']:
            topic_id = item['topic_id'] + 1
            html_content += f"""
    <div class="topic" id="topic-{topic_id}">
        <h2>Тема {topic_id}: {item['title']}</h2>
"""
            
            if item['keywords']:
                html_content += '        <div class="keywords"><strong>Ключевые слова:</strong> '
                html_content += ", ".join(item['keywords'])
                html_content += "</div>\n"
            
            html_content += f"""
        <div class="content">
            <p>{item['text'].replace(chr(10), '</p><p>')}</p>
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _create_wordcloud(self, doc: Dict, output_path: Path) -> None:
        """Создание облака слов для всего документа"""
        # Объединение всего текста
        all_text = " ".join([item['text'] for item in doc['content']])
        
        # Создание облака слов
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            normalize_plurals=False
        ).generate(all_text)
        
        # Сохранение изображения
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_topic_visualization(self, doc: Dict, output_path: Path) -> None:
        """Создание визуализации тем"""
        # Подсчет количества предложений для каждой темы
        topic_counts = {}
        for item in doc['content']:
            topic_id = item['topic_id']
            sentence_count = len(item['text'].split('.'))
            topic_counts[topic_id] = sentence_count
        
        # Создание DataFrame для визуализации
        df = pd.DataFrame({
            'Topic': [f"Тема {topic_id + 1}" for topic_id in topic_counts.keys()],
            'Sentences': list(topic_counts.values())
        })
        
        # Создание графика
        plt.figure(figsize=(10, 6))
        plt.bar(df['Topic'], df['Sentences'], color='skyblue')
        plt.title('Распределение предложений по темам')
        plt.xlabel('Темы')
        plt.ylabel('Количество предложений')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Основная функция для запуска обработки транскрипций"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    processor = TranscriptProcessor()
    processor.process_all_transcripts()
    
    logging.info("Обработка всех транскрипций завершена!")

if __name__ == "__main__":
    main()
```

### Интеграция с вашим проектом

Чтобы интегрировать этот скрипт с вашим проектом, вам нужно:

1. Добавить новые зависимости в `requirements.txt`:
```
sentence-transformers
scikit-learn
nltk
matplotlib
wordcloud
pandas
```

2. Создать новый скрипт `process_transcripts.py` в корневой директории проекта с кодом выше.

3. Добавить запуск обработки транскрипций в основной пайплайн. Для этого можно модифицировать `transcribe_v2.py`, добавив в конец функции `process_files` вызов нового скрипта:

```python
# В конце функции process_files в transcribe_v2.py
if config.process_transcripts:
    logger.info("Запуск пост-обработки транскрипций...")
    try:
        from process_transcripts import TranscriptProcessor
        processor = TranscriptProcessor(
            input_dir=config.transcripts_out,
            output_dir=config.processed_transcripts_out
        )
        processor.process_all_transcripts()
        logger.info("Пост-обработка транскрипций завершена")
    except Exception as exc:
        logger.error("Ошибка при пост-обработке транскрипций: %s", exc)
```

4. Добавить новые параметры в конфигурацию (`.env`):
```
# Обработка транскрипций
PROCESS_TRANSCRIPTS=true
PROCESSED_TRANSCRIPTS_OUT=processed_transcripts
TRANSCRIPT_MODEL=all-MiniLM-L6-v2
TRANSCRIPT_MAX_TOPICS=10
TRANSCRIPT_MIN_CLUSTER_SIZE=5
```

5. Обновить класс `PipelineConfig` в `transcribe_v2.py`:

```python
# В классе PipelineConfig добавить новые поля
process_transcripts: bool = True
processed_transcripts_out: Path = Path("processed_transcripts")
transcript_model: str = "all-MiniLM-L6-v2"
transcript_max_topics: int = 10
transcript_min_cluster_size: int = 5

# В методе from_env добавить чтение новых переменных окружения
process_transcripts=_parse_bool(os.getenv("PROCESS_TRANSCRIPTS"), True),
processed_transcripts_out=Path(os.getenv("PROCESSED_TRANSCRIPTS_OUT", "processed_transcripts")),
transcript_model=os.getenv("TRANSCRIPT_MODEL", "all-MiniLM-L6-v2"),
transcript_max_topics=_parse_int(os.getenv("TRANSCRIPT_MAX_TOPICS"), 10),
transcript_min_cluster_size=_parse_int(os.getenv("TRANSCRIPT_MIN_CLUSTER_SIZE"), 5),
```

### Результаты

После выполнения этого скрипта вы получите для каждого вебинара:

1. **Структурированный документ** с разделением на темы
2. **Оглавление** для быстрой навигации
3. **Ключевые слова** для каждой темы
4. **Резюме** вебинара
5. **Облако слов** для визуального представления主要内容
6. **График распределения тем** для понимания структуры вебинара

Документы будут сохранены в нескольких форматах:
- JSON (для программной обработки)
- Markdown (для чтения и редактирования)
- HTML (для просмотра в браузере)

### Дополнительные возможности

Для дальнейшего улучшения можно добавить:

1. **Поиск по семантике** с использованием векторной базы данных (например, ChromaDB):
```python
import chromadb
from chromadb.config import Settings

class SemanticSearch:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(settings=Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection("transcripts")
    
    def add_transcript(self, doc_id: str, title: str, content: Dict):
        # Добавление документа в векторную базу
        pass
    
    def search(self, query: str, n_results: int = 5):
        # Поиск по запросу
        pass
```

2. **Создание интерактивного веб-интерфейса** с использованием Streamlit:
```python
import streamlit as st
from process_transcripts import TranscriptProcessor
from semantic_search import SemanticSearch

def main():
    st.title("Поиск по вебинарам")
    
    # Инициализация поиска
    search = SemanticSearch()
    
    # Поле для ввода запроса
    query = st.text_input("Введите поисковый запрос:")
    
    if query:
        results = search.search(query)
        # Отображение результатов
        pass

if __name__ == "__main__":
    main()
```

3. **Автоматическое создание кратких выжимок** для каждой темы с использованием моделей суммаризации:
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="IlyaGusev/rut5_base_sum_gazeta")

def summarize_text(text: str, max_length: int = 150) -> str:
    """Создание краткой выжимки для текста"""
    result = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    return result[0]['summary_text']
```

### Заключение

С этим решением вы сможете превратить ваши 300+ вебинаров в структурированную, легко доступную базу знаний. Вы сможете быстро находить нужную информацию, получать краткие выжимки и эффективно использовать накопленный контент.

Не волнуйтесь, ваши записи не пропали зря - с правильной обработкой они станут ценным ресурсом, который будет служить вам долгие годы!

Если у вас есть вопросы или нужна помощь с реализацией, не стесняйтесь обращаться.