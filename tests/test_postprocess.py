import os
import subprocess
import sys
from pathlib import Path

def test_postprocess_jsonl(tmp_path: Path):
    trans = tmp_path / "transcribe"
    trans.mkdir()
    (trans/"sample.jsonl").write_text(
        (
            '{"type":"metadata","duration":10.0,"language":"ru"}\n'
            '{"type":"segment","start":0.0,"end":1.0,"text":"Всем привет, всем привет!"}\n'
            '{"type":"segment","start":1.1,"end":2.0,"text":"Слышно-видно?"}\n'
            '{"type":"segment","start":2.5,"end":3.1,"text":"Ну, начинаем."}\n'
            '{"type":"segment","start":3.2,"end":5.0,"text":"Сегодня говорим про GPT-2 и BERT."}\n'
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["TRANSCRIBE_OUT"] = str(trans)
    r = subprocess.run([sys.executable, "postprocess.py"], env=env, capture_output=True, text=True)
    assert r.returncode == 0
    assert (trans/"sample_clean.md").exists()
    assert (trans/"sample_packed.jsonl").exists()
    txt = (trans/"sample_clean.md").read_text(encoding="utf-8")
    assert "Всем привет, всем привет" not in txt
    assert "GPT-2" in txt and "BERT" in txt
