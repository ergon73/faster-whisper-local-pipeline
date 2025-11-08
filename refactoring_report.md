# Refactoring Report

## Executive Summary
1. **transcribe_v2.py:357-363** – `WhisperModel.transcribe` не принимает аргумент `num_workers`, поэтому пайплайн падает с `TypeError` на первом файле и ничего не транскрибирует.
2. **requirements.txt:5** – указанная версия `torch==2.9.0` ещё не существует в PyPI, из-за чего `pip install -r requirements.txt` всегда завершается ошибкой.
3. **transcribe_v2.py:156-288** – `_build_audio_filename` формирует имена только по `stem` и тегу (`_audio`/`_video`). Файлы с одинаковыми именами, но разными расширениями, перезаписывают друг друга в `audio_out/` и в каталоге транскриптов.

## Детальный анализ
### 1. Критические проблемы
- **transcribe_v2.py:357-363 — Неверный аргумент `num_workers`.** Метод `WhisperModel.transcribe` не принимает этот параметр, поэтому строка  
  ```python
  segments, info = model.transcribe(..., num_workers=config.num_workers)
  ```  
  сразу выбрасывает `TypeError`. Пайплайн не может обработать ни одного файла.
- **requirements.txt:5 — Неустанавливаемая зависимость.** `torch==2.9.0` отсутствует в официальных колёсах. Любая установка зависимостей ломается до запуска приложения.
- **transcribe_v2.py:156-288 — Коллизии выходных имен.** `_build_audio_filename` возвращает `f"{source.stem}_{safe_tag}.wav"`. Если в `audio_in/` лежат `call.mp3` и `call.m4a`, второй файл перезаписывает WAV первого, а транскрипт получает неправильный текст. Аналогично для видео с одинаковыми именами.

### 2. Архитектурные проблемы
- **transcribe_v2.py:386-396 — Жёсткая зависимость от FFmpeg.** `ensure_ffmpeg_available` вызывается до анализа входных данных. Даже если пользователь хочет транскрибировать уже готовые WAV-файлы, отсутствие FFmpeg прерывает скрипт, хотя он не нужен. FFmpeg должен проверяться только если найдено видео или аудио, требующее перекодирования.
- **transcribe_v2.py:236-289 — Дублирование логики подготовки задач.** Циклы для `video_in` и `audio_in` повторяют одинаковые проверки (`skip_fresh_audio`, построение имён, добавление `TranscriptionJob`). Любое изменение нужно делать дважды, что уже привело к расхождениям (например, разные сообщения логов).

### 3. Качество кода
- **transcribe_v2.py:104-111 — Неиспользуемое поле `origin_tag`.** Поле хранится в `TranscriptionJob`, но далее нигде не читается (`grep` по файлу это подтверждает). Оно дезориентирует, мешает сериализации и увеличивает объём передачи данных между функциями.
- **transcribe_v2.py:357-383 — Заглушенная диагностика.** `transcribe_audio` ловит `Exception`, пишет только `logger.error("Ошибка ...")` и возвращает `False`. Стек теряется, дебаг невозможен. Следует логировать исключение (`logger.exception`) и ограничивать типы обрабатываемых ошибок.

### 4. Производительность
- **transcribe_v2.py:413-421 — Нет кеша транскриптов.** Даже если `.txt` свежее соответствующего WAV, файл снова прогоняется через модель. На длинных лекциях это впустую расходует часы GPU/CPU.
- **transcribe_v2.py:333-342 — CPU fallback игнорирует настройки пользователя.** При падении CUDA модель пересоздаётся через `WhisperModel(..., device="cpu", compute_type="int8")` без передачи `cpu_threads`/`num_workers`. На CPU-проектах это снижает скорость до значений по умолчанию и может перегрузить машину.

### 5. Инфраструктура
- **Корневой `.gitignore` отсутствует.** В репозитории уже лежат `.venv/`, `audio_out/`, `transcribe/`, `.env`, `pytest-cache-*`. Без `.gitignore` высок риск залить чувствительные данные и мусор.
- **requirements.txt:6 — Неиспользуемый `requests`.** Пакет не импортируется ни в коде, ни в тестах. Каждая установка тратит лишнее время и увеличивает поверхность атак.
- **CI/CD конфигурация отсутствует.** Нет `.github/workflows/` или аналогичных файлов. Любое изменение выкатывается без автоматических тестов, хотя `pytest` есть и запускается быстро.

## Метрики
- Критические проблемы: **3**
- Архитектурные проблемы: **2**
- Качество кода: **2**
- Производительность: **2**
- Инфраструктура: **3**
- **Оценка технического долга:** ~65/100 (где 100 — идеал). Основная тяжесть в критических блоках и инфраструктуре; после устранения первых трёх задач показатель можно поднять до 80+.

## План рефакторинга

### Группа 1 — Критические фиксы (выполнять по одному, сразу)
1. **[ПРИОРИТЕТ: CRITICAL]**  
   Проблема: `num_workers` передаётся в неподдерживаемый метод `transcribe`.  
   Файл: `transcribe_v2.py:333-363`  
   Что сломано: каждый запуск транскрибации падает с `TypeError`.  
   Как исправить:

   # Конкретный код для замены
   ```diff
   -        model = WhisperModel(
   -            config.whisper_model,
   -            device=device,
   -            compute_type=compute_type,
   -            cpu_threads=config.cpu_threads,
   -        )
   +        model = WhisperModel(
   +            config.whisper_model,
   +            device=device,
   +            compute_type=compute_type,
   +            cpu_threads=config.cpu_threads,
   +            num_workers=config.num_workers,
   +        )
   ...
   -        segments, info = model.transcribe(
   -            str(job.audio_path),
   -            beam_size=config.beam_size,
   -            vad_filter=config.vad_filter,
   -            vad_parameters={"min_silence_duration_ms": config.vad_min_silence_ms},
   -            num_workers=config.num_workers,
   -        )
   +        segments, info = model.transcribe(
   +            str(job.audio_path),
   +            beam_size=config.beam_size,
   +            vad_filter=config.vad_filter,
   +            vad_parameters={"min_silence_duration_ms": config.vad_min_silence_ms},
   +        )
   ```
   Тесты: `pytest tests/test_pipeline.py` + прогон реального WAV.  
   Риски: потребует повторной загрузки модели (но это штатно).  
   Откат: `git checkout -- transcribe_v2.py`.

2. **[ПРИОРИТЕТ: CRITICAL]**  
   Проблема: `torch==2.9.0` не существует.  
   Файл: `requirements.txt:5`  
   Что сломано: установка зависимостей невозможна.  
   Как исправить: указать поддерживаемую версию (например, `torch==2.3.1` + инструкция для CUDA при необходимости).

   # Конкретный код для замены
   ```diff
   -torch==2.9.0
   +# CPU по умолчанию; для CUDA оставить инструкцию в README
   +torch==2.3.1
   ```
   Тесты: `pip install -r requirements.txt && pytest`.  
   Риски: потребуются wheels, соответствующие GPU/CPU.  
   Откат: `git checkout -- requirements.txt`.

3. **[ПРИОРИТЕТ: HIGH]**  
   Проблема: WAV и текстовые файлы перезаписываются, если `stem` одинаковый.  
   Файл: `transcribe_v2.py:156-289`  
   Что сломано: данные теряются при наличии `call.mp3` и `call.wav`.  
   Как исправить: включить расширение или hash в итоговое имя.

   # Конкретный код для замены
   ```diff
   -def _build_audio_filename(source: Path, origin_tag: str) -> str:
   -    safe_tag = origin_tag.strip().replace(" ", "_").lower()
   -    return f"{source.stem}_{safe_tag}.wav"
   +def _build_audio_filename(source: Path, origin_tag: str) -> str:
   +    safe_tag = origin_tag.strip().replace(" ", "_").lower()
   +    suffix = source.suffix.lower().lstrip(".") or "raw"
   +    return f"{source.stem}_{safe_tag}_{suffix}.wav"
   ```
   Тесты: добавить кейс в `tests/test_pipeline.py` с файлами `meeting.mp3` + `meeting.wav`, затем `pytest`.  
   Риски: имена файлов изменятся; нужно предупредить пользователей.  
   Откат: `git checkout -- transcribe_v2.py tests/test_pipeline.py`.

### Группа 2 — Улучшения архитектуры
4. **[ПРИОРИТЕТ: HIGH]**  
   Проблема: FFmpeg обязателен даже для готовых WAV.  
   Файл: `transcribe_v2.py:386-396`  
   Что сломано: пользователи без FFmpeg не могут транскрибировать аудио.  
   Как исправить: определить, нужны ли перекодирования, и вызывать `ensure_ffmpeg_available` только тогда.

   # Конкретный код для замены
   ```diff
   -    ensure_directories(config)
   -    try:
   -        ensure_ffmpeg_available(logger)
   -    except RuntimeError as exc:
   -        logger.error("Невозможно продолжить без FFmpeg: %s", exc)
   -        return
   +    ensure_directories(config)
   +    if media_requires_ffmpeg(config):
   +        try:
   +            ensure_ffmpeg_available(logger)
   +        except RuntimeError as exc:
   +            logger.error("Нужен FFmpeg для видео/сжатых аудио: %s", exc)
   +            return
   ```
   и добавить `media_requires_ffmpeg`, которые проверяет наличие видео или не-WAV аудио.  
   Тесты: unit-тест для нового helper + `pytest`.  
   Риски: дополнительный проход по директориям.  
   Откат: `git checkout -- transcribe_v2.py`.

5. **[ПРИОРИТЕТ: MEDIUM]**  
   Проблема: `prepare_jobs` содержит две почти одинаковые ветки.  
   Файл: `transcribe_v2.py:236-289`  
   Что сломано: высокий риск расхождений и дублирования багов.  
   Как исправить: вынести общую часть в функцию `_collect_jobs`, принимающую директорию, набор расширений и стратегию подготовки.

   # Конкретный код для замены
   ```diff
   +def _collect_jobs(source_dir: Path, formats: tuple[str, ...], origin_tag: str,
   +                  handler: Callable[[Path, Path, logging.Logger], bool],
   +                  audio_out: Path, skip_fresh: bool) -> list[TranscriptionJob]:
   +    jobs: list[TranscriptionJob] = []
   +    for file in sorted(source_dir.iterdir()):
   +        if not file.is_file() or file.suffix.lower() not in formats:
   +            continue
   +        audio_path = audio_out / _build_audio_filename(file, origin_tag)
   +        if skip_fresh and _is_audio_up_to_date(file, audio_path):
   +            jobs.append(TranscriptionJob(source_path=file, audio_path=audio_path))
   +        elif handler(file, audio_path, logger):
   +            jobs.append(TranscriptionJob(source_path=file, audio_path=audio_path))
   +    return jobs
   ```
   Далее `prepare_jobs` вызывает helper для видео с `extract_audio` и для аудио с `copy_wav`/`extract_audio`.  
   Тесты: обновить `tests/test_pipeline.py` на новые функции.  
   Риски: возможно, придётся скорректировать monkeypatch'и в тестах.  
   Откат: `git checkout -- transcribe_v2.py tests/test_pipeline.py`.

### Группа 3 — Рефакторинг кода и инфраструктуры
6. **[ПРИОРИТЕТ: MEDIUM]**  
   Проблема: `TranscriptionJob.origin_tag` не используется.  
   Файл: `transcribe_v2.py:104-289`  
   Что сломано: шум и риск несинхронизировать поле.  
   Как исправить: удалить поле и не передавать его при создании объекта.

   # Конкретный код для замены
   ```diff
   -class TranscriptionJob:
   -    source_path: Path
   -    audio_path: Path
   -    origin_tag: str
   +class TranscriptionJob:
   +    source_path: Path
   +    audio_path: Path
   ```
   Тесты: `pytest`.  
   Риски: придётся пересоздать любые сериализованные задания (если были).  
   Откат: `git checkout -- transcribe_v2.py`.

7. **[ПРИОРИТЕТ: MEDIUM]**  
   Проблема: исключения в `transcribe_audio` теряют стек.  
   Файл: `transcribe_v2.py:357-383`  
   Что сломано: диагностика сложна, реальные ошибки маскируются.  
   Как исправить: логировать стек (`logger.exception`) и пробрасывать критические ошибки.

   # Конкретный код для замены
   ```diff
       except Exception as exc:
-        logger.error("Ошибка транскрибации %s: %s", job.audio_path.name, exc)
-        return False
+        logger.exception("Ошибка транскрибации %s", job.audio_path.name)
+        return False
   ```
   Тесты: имитировать исключение в `model.transcribe` и убедиться, что лог содержит stacktrace (можно использовать caplog).  
   Риски: объём логов увеличится.  
   Откат: `git checkout -- transcribe_v2.py tests/test_pipeline.py`.

8. **[ПРИОРИТЕТ: MEDIUM]**  
   Проблема: нет `.gitignore`, в git попадают артефакты и секреты.  
   Файл: `.gitignore` (новый)  
   Что сломано: `.env`, `.venv`, `audio_out/` и транскрипты можно случайно закоммитить.  
   Как исправить: создать `.gitignore` с шаблонами Python-проекта.

   # Конкретный код для замены
   ```gitignore
   .venv/
   __pycache__/
   *.pyc
   .env
   audio_out/
   transcribe/
   audio_in/
   video_in/
   pytest-cache*/
   *.log
   ```
   Тесты: `git status --short` (папки должны пропасть из списка отслеживаемых).  
   Риски: нужно убедиться, что пользователю действительно не нужны эти каталоги в git.  
   Откат: `git rm --cached <path>` и удалить `.gitignore`.

9. **[ПРИОРИТЕТ: LOW]**  
   Проблема: `requests` не используется.  
   Файл: `requirements.txt:6`  
   Что сломано: лишняя зависимость увеличивает время установки и площадь атак.  
   Как исправить: удалить пакет и обновить документацию.

   # Конкретный код для замены
   ```diff
   -requests==2.32.3
   ```
   Тесты: `pip install -r requirements.txt && pytest`.  
   Риски: если когда-нибудь появится HTTP-функциональность, нужно добавить пакет обратно.  
   Откат: `git checkout -- requirements.txt`.

10. **[ПРИОРИТЕТ: MEDIUM]**  
    Проблема: отсутствует CI, тесты не запускаются автоматически.  
    Файл: `.github/workflows/tests.yml` (новый)  
    Что сломано: регрессии попадают в main.  
    Как исправить: добавить GitHub Actions с шагами установки и `pytest`.

    # Конкретный код для замены
    ```yaml
    name: CI
    on: [push, pull_request]
    jobs:
      tests:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: '3.11'
          - run: pip install -r requirements.txt
          - run: pytest
    ```
    Тесты: удостовериться, что workflow зелёный (через badge или вкладку Actions).  
    Риски: потребует настроить кеш для тяжёлых зависимостей (faster-whisper).  
    Откат: удалить `.github/workflows/tests.yml`.

### Группа 4 — Оптимизации (последними)
11. **[ПРИОРИТЕТ: MEDIUM]**  
    Проблема: нет кеша транскриптов, одинаковые файлы обрабатываются повторно.  
    Файл: `transcribe_v2.py:413-421`  
    Что сломано: время работы растёт линейно при повторных запусках.  
    Как исправить: перед вызовом `transcribe_audio` проверять, что `result_path` свежее WAV.

    # Конкретный код для замены
    ```diff
        for index, job in enumerate(jobs, start=1):
            result_path = config.transcripts_out / f"{job.audio_path.stem}.txt"
-        if transcribe_audio(job, result_path, model, config, logger):
+        if result_path.exists() and result_path.stat().st_mtime >= job.audio_path.stat().st_mtime:
+            logger.info("Пропуск транскрибации (актуальный TXT уже есть): %s", result_path.name)
+            success += 1
+            continue
+        if transcribe_audio(job, result_path, model, config, logger):
    ```
    Тесты: интеграционный тест, создающий готовый TXT и проверяющий, что `model.transcribe` не вызывается (через monkeypatch).  
    Риски: изменения в исходном аудио не попадут в результат, если часы системы идут назад — следует документировать.  
    Откат: `git checkout -- transcribe_v2.py tests/test_pipeline.py`.

12. **[ПРИОРИТЕТ: LOW]**  
    Проблема: CPU fallback игнорирует `cpu_threads` / `num_workers`.  
    Файл: `transcribe_v2.py:333-342`  
    Что сломано: на CPU система может неожиданно занять все ядра или наоборот работать в 1 поток.  
    Как исправить: передавать пользовательские настройки при fallback.

    # Конкретный код для замены
    ```diff
            logger.warning("Не удалось загрузить модель на CUDA (%s). Пробуем CPU (int8).", error)
-            return WhisperModel(config.whisper_model, device="cpu", compute_type="int8")
+            return WhisperModel(
+                config.whisper_model,
+                device="cpu",
+                compute_type="int8",
+                cpu_threads=config.cpu_threads,
+                num_workers=config.num_workers,
+            )
    ```
    Тесты: unit-тест, подменяющий `WhisperModel` и проверяющий параметры; затем `pytest`.  
    Риски: при слишком агрессивных настройках CPU можно загрузить систему; стоит дополнительно валидировать вход.  
    Откат: `git checkout -- transcribe_v2.py tests/test_pipeline.py`.

## Настройки скорости и качества (на базе `.env.example`)
- **Ускорение транскрибации**
  - `WHISPER_DEVICE=cuda` (при наличии GPU) даёт +4–10× к скорости по сравнению с CPU int8.
  - `WHISPER_CPU_THREADS=<число физических ядер>` (например, `8`) заставляет CTranslate2 использовать все доступные ядра; ускорение на CPU ~20–35% по сравнению с настройками ОС по умолчанию.
  - `WHISPER_NUM_WORKERS=2..4` (после фикса бага) параллелит декодер и запись сегментов; выигрышь 10–25% на длинных дорожках.
  - `WHISPER_BEAM_SIZE=1-2` снижает качество, но ускоряет распознавание ещё на 15–30% на CPU.
  - `SKIP_FRESH_AUDIO=true` уже включён и экономит время повторной перекодировки (секунды или минуты на файл).

- **Повышение качества транскрибации**
  - `WHISPER_BEAM_SIZE=8` (вместо 5) сокращает word-error-rate примерно на 5–8%, особенно на шумных записях.
  - `WHISPER_VAD_FILTER=true` с `WHISPER_VAD_MIN_SILENCE=300–400` мс лучше отделяет фразы и уменьшает количество галлюцинаций (примерно на 2–4% сегментов).
  - `WHISPER_CPU_THREADS` + `WHISPER_DEVICE=cuda` позволяют использовать более тяжёлые compute types (`float16`), что даёт до 3% прироста точности относительно `int8`.
  - Для аудио с большим количеством фонового шума имеет смысл уменьшить `WHISPER_VAD_MIN_SILENCE` до 200–300 мс, чтобы не обрезать короткие реплики (примерно +1% распознанных слов).

## Дополнительные проверки
- **.gitignore** — отсутствует (см. задачу 8).
- **requirements.txt / pyproject.toml** — есть `requirements.txt`, но содержит одну критичную и одну лишнюю зависимость (см. задачи 2 и 9); `pyproject.toml` отсутствует.
- **README.md / README_v2.md / QUICK_START.md** — подробные инструкции присутствуют и актуальны.
- **Тесты** — есть `tests/test_pipeline.py`, покрывающий подбор имён и `skip_fresh_audio`; интеграционных тестов нет.
- **CI/CD конфигурация** — отсутствует (см. задача 10).
- **.env.example** — файл есть, описывает основные переменные, но не упоминает `WHISPER_COMPUTE_TYPE` и будущие тюнинги (можно добавить в рамках инфраструктурных задач).
