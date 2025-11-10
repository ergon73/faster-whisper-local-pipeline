"""Пост-обработка транскриптов Faster-Whisper в чистые абзацы."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

DEFAULT_TRANSCRIBE_OUT = "transcribe"

def getenv_str(key: str, default: str) -> str:
    """Возвращает строку из переменной окружения или значение по умолчанию."""
    v = os.getenv(key)
    return v if v is not None and v.strip() else default

def getenv_int(key: str, default: int) -> int:
    """Возвращает целое число из окружения с запасным значением."""
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def getenv_bool(key: str, default: bool) -> bool:
    """Интерпретирует булевые переменные окружения."""
    v = os.getenv(key)
    if v is None: return default
    return v.strip().lower() in {"1","true","yes","y","on"}

SRT_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
def _parse_srt_ts(s: str) -> Optional[float]:
    """Преобразует строковый таймкод SRT в секунды."""
    m = SRT_TS.search(s)
    if not m: return None
    hh, mm, ss, ms = map(int, m.groups())
    return hh*3600 + mm*60 + ss + ms/1000.0

@dataclass
class Segment:
    """Сегмент исходной транскрибации."""
    start: Optional[float]
    end: Optional[float]
    text: str

@dataclass
class Paragraph:
    """Объединённый абзац из одного или нескольких сегментов."""
    id: int
    start: Optional[float]
    end: Optional[float]
    text: str
    segments: List[Segment]

def iter_segments_from_jsonl(path: Path) -> Tuple[dict, List[Segment]]:
    """Считывает метаданные и сегменты из JSONL."""
    meta = {}
    segs: List[Segment] = []
    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            t = row.get("type")
            if first and t == "metadata":
                meta = row; first = False; continue
            if t in {"segment", "Segment"}:
                start = row.get("start"); end = row.get("end")
                text = (row.get("text") or "").strip()
                if text:
                    segs.append(Segment(
                        start=float(start) if start is not None else None,
                        end=float(end) if end is not None else None,
                        text=text
                    ))
            first = False
    return meta, segs

def iter_segments_from_srt(path: Path) -> Tuple[dict, List[Segment]]:
    """Считывает сегменты из SRT, извлекая таймкоды и текст."""
    meta = {}; segs: List[Segment] = []; block: List[str] = []
    def flush(block: List[str]):
        if len(block) < 2: return
        ts = block[0] if "-->" in block[0] else (block[1] if len(block)>1 and "-->" in block[1] else "")
        start = _parse_srt_ts(ts.split("-->")[0]) if "-->" in ts else None
        end   = _parse_srt_ts(ts.split("-->")[1]) if "-->" in ts else None
        text_lines = []
        for line in block:
            if "-->" in line: continue
            if line.strip().isdigit(): continue
            text_lines.append(line)
        text = " ".join([t.strip() for t in text_lines]).strip()
        if text: segs.append(Segment(start, end, text))
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                flush(block); block = []
            else:
                block.append(line)
        flush(block)
    return meta, segs

def iter_segments_from_txt(path: Path) -> Tuple[dict, List[Segment]]:
    """Создаёт сегменты из TXT без таймингов."""
    meta = {"no_timings": True}; segs: List[Segment] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t: segs.append(Segment(None, None, t))
    return meta, segs

PUNCT_STRONG = set(".?!…")
def normalize_text(s: str) -> str:
    """Нормализует пробелы и повторяющуюся пунктуацию."""
    s = re.sub(r"\s+", " ", s).strip()
    if "...." in s or ".." in s:
        s = s.replace("....", "...")
    s = re.sub(r"([!?.,])\1{2,}", r"\1\1", s)
    return s

def strip_fillers_edge(s: str, fillers: list[str]) -> str:
    """Удаляет указанные филлеры по краям строки."""
    if not fillers: return s
    for _ in range(2):
        for token in fillers:
            head_pattern = rf"^(?:{re.escape(token)})(?:[\s,;:!?.…-]+)"
            tail_pattern = rf"(?:[\s,;:!?.…-]+)(?:{re.escape(token)})(?:[\s!?.…,-]*)$"
            s = re.sub(head_pattern, "", s, flags=re.IGNORECASE)
            s = re.sub(tail_pattern, "", s, flags=re.IGNORECASE)
    return s

def is_strong_end(s: str) -> bool:
    """Проверяет, оканчивается ли строка сильной пунктуацией."""
    return len(s) > 0 and s[-1] in PUNCT_STRONG

def almost_equal(a: str, b: str, threshold: float = 0.9) -> bool:
    """Оценивает близость строк по SequenceMatcher."""
    from difflib import SequenceMatcher
    if len(a) > 60 or len(b) > 60: return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def dedupe_runs(lines: list[str], strict: bool, fuzzy: bool) -> list[str]:
    """Удаляет подряд идущие дубликаты и почти-дубликаты строк."""
    if not lines: return lines
    result = [lines[0]]
    for line in lines[1:]:
        prev = result[-1]
        if strict and line.strip().lower() == prev.strip().lower(): continue
        if fuzzy and almost_equal(line, prev): continue
        result.append(line)
    return result

def merge_segments_to_paragraphs(segs: list[Segment], gap_ms: int, min_chars: int, max_par_len: int,
                                 fillers_ru: list[str], fillers_en: list[str]) -> list[dict]:
    """Склеивает сегменты в абзацы с учётом пауз, длины и финальной пунктуации."""
    def gap(prev: Segment, cur: Segment) -> Optional[float]:
        if prev.end is None or cur.start is None: return None
        return (cur.start - prev.end) * 1000.0
    out: list[dict] = []
    current = None
    pid = 0
    for s in segs:
        t = normalize_text(s.text)
        t = strip_fillers_edge(t, fillers_ru)
        t = strip_fillers_edge(t, fillers_en)
        if not t: continue
        s_clean = Segment(s.start, s.end, t)
        if current is None:
            pid += 1
            current = {"id": pid, "start": s_clean.start, "end": s_clean.end, "text": s_clean.text, "segments":[s_clean]}
            continue
        g = gap(current["segments"][-1], s_clean)
        should_merge = (g is not None and g <= gap_ms) or (len(current["text"]) < min_chars) or (not is_strong_end(current["text"])) or (len(s_clean.text) < 15)
        candidate = (current["text"] + " " + s_clean.text).strip() if should_merge else None
        if should_merge and candidate and len(candidate) <= max_par_len:
            current["text"] = candidate
            current["end"] = s_clean.end if s_clean.end is not None else current["end"]
            current["segments"].append(s_clean)
        else:
            out.append(current)
            pid += 1
            current = {"id": pid, "start": s_clean.start, "end": s_clean.end, "text": s_clean.text, "segments":[s_clean]}
    if current is not None: out.append(current)
    return out

def write_packed_jsonl(pars: list[dict], meta: dict, path: Path) -> None:
    """Сохраняет абзацы и метаданные в JSONL."""
    meta_out = dict(meta); meta_out["granularity"] = "paragraph"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type":"metadata", **meta_out}, ensure_ascii=False) + "\n")
        for p in pars:
            row = {"type":"paragraph","id":p["id"],"text":p["text"],"segments":[{"start": s.start, "end": s.end, "text": s.text} for s in p["segments"]]}
            if p.get("start") is not None: row["start"] = p["start"]
            if p.get("end")   is not None: row["end"]   = p["end"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_clean_md(pars: list[dict], path: Path) -> None:
    """Сохраняет очищенные абзацы в Markdown."""
    with path.open("w", encoding="utf-8") as f:
        for p in pars: f.write(p["text"].strip() + "\n\n")

def write_clean_txt(pars: list[dict], path: Path) -> None:
    """Сохраняет очищенный текст в TXT."""
    with path.open("w", encoding="utf-8") as f:
        for p in pars: f.write(p["text"].strip() + "\n")

def write_report(report: dict, path: Path) -> None:
    """Сохраняет отчёт о постобработке."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

def main() -> None:
    """CLI-точка входа для пост-обработки транскриптов."""
    if load_dotenv is not None: load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("postprocess")

    out_dir = Path(getenv_str("TRANSCRIBE_OUT", DEFAULT_TRANSCRIBE_OUT))
    if not out_dir.exists():
        logger.error("Directory not found: %s", out_dir); sys.exit(1)

    cfg = {
        "min_chars": getenv_int("POST_MIN_CHARS", 48),
        "merge_gap_ms": getenv_int("POST_MERGE_GAP_MS", 900),
        "max_par_len": getenv_int("POST_MAX_PAR_LEN", 1200),
        "strict_dedup": getenv_bool("POST_STRICT_DEDUP", True),
        "fuzzy_dedup": getenv_bool("POST_FUZZY_DEDUP", True),
        "write_clean_txt": getenv_bool("POST_WRITE_CLEAN_TXT", False),
        "write_report": getenv_bool("POST_REPORT", True),
        "fillers_ru": [
            s.strip()
            for s in getenv_str(
                "POST_FILLERS_RU",
                "ну,как бы,то есть,ээ,эээ,короче,значит,в общем,всем привет,слышно видно,слышно-видно",
            ).split(",")
            if s.strip()
        ],
        "fillers_en": [
            s.strip()
            for s in getenv_str(
                "POST_FILLERS_EN",
                "uh,um,like,you know,so,kinda,sort of,hi everyone,can you hear me",
            ).split(",")
            if s.strip()
        ],
    }

    # собрать уникальные stem'ы (каталог + базовое имя без расширения)
    stems: set[tuple[Path, str]] = set()
    for p in out_dir.iterdir():
        if p.suffix.lower() in {".jsonl", ".srt", ".txt"}:
            base_name = p.name.rsplit(".", 1)[0]
            stems.add((p.parent, base_name))

    if not stems:
        logger.warning("No input files found in %s", out_dir); return

    for parent, base_name in sorted(stems, key=lambda item: item[1]):
        src_jsonl = parent / f"{base_name}.jsonl"
        src_srt = parent / f"{base_name}.srt"
        src_txt = parent / f"{base_name}.txt"
        source, meta, segs = None, {}, []

        if src_jsonl.exists():
            meta, segs = iter_segments_from_jsonl(src_jsonl); source = "jsonl"
        elif src_srt.exists():
            meta, segs = iter_segments_from_srt(src_srt); source = "srt"
        elif src_txt.exists():
            meta, segs = iter_segments_from_txt(src_txt); source = "txt"
        else:
            logger.info("Skip %s (no inputs)", base_name); continue

        raw_lines = [s.text for s in segs]
        lines_dedup = dedupe_runs(raw_lines, strict=cfg["strict_dedup"], fuzzy=cfg["fuzzy_dedup"])

        if len(lines_dedup) != len(raw_lines):
            # упрощённая реконструкция соответствия (при строгом дедупе равенство сохраняется чаще всего)
            new_segs = []
            it = iter(lines_dedup); nxt = next(it, None)
            for s in segs:
                if nxt is None: break
                if s.text.strip() == nxt.strip():
                    new_segs.append(s); nxt = next(it, None)
            segs = new_segs if new_segs else segs  # на случай fuzzy несовпадений

        pars = merge_segments_to_paragraphs(
            segs,
            gap_ms=cfg["merge_gap_ms"],
            min_chars=cfg["min_chars"],
            max_par_len=cfg["max_par_len"],
            fillers_ru=cfg["fillers_ru"],
            fillers_en=cfg["fillers_en"],
        )

        packed_path = parent / f"{base_name}_packed.jsonl"
        clean_md_path = parent / f"{base_name}_clean.md"
        clean_txt_path = parent / f"{base_name}_clean.txt"
        report_path = parent / f"{base_name}_postprocess_report.json"

        write_packed_jsonl(pars, meta, packed_path)
        write_clean_md(pars, clean_md_path)
        if cfg["write_clean_txt"]:
            write_clean_txt(pars, clean_txt_path)

        if cfg["write_report"]:
            report = {
                "source": source,
                "no_timings": bool(meta.get("no_timings", False)),
                "paragraphs": len(pars),
                "avg_par_len": (sum(len(p['text']) for p in pars) / max(1,len(pars))),
                "config": {k:v for k,v in cfg.items() if k not in {"fillers_ru","fillers_en"}},
            }
            write_report(report, report_path)

        logger.info("Processed %s → %s / %s", base_name, clean_md_path.name, packed_path.name)

if __name__ == "__main__":
    main()