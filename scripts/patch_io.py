from pathlib import Path

io_path = Path("src/legalqa/core/io.py")
text = io_path.read_text(encoding="utf-8")
if "def sha256_file" not in text:
    extra = """

def sha256_file(path: str | Path) -> str:
    import hashlib
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
"""
    io_path.write_text(text + extra, encoding="utf-8")
    print("Added sha256_file and write_json to src/legalqa/core/io.py")
