# ingestion/loader.py

from pathlib import Path
from typing import Iterator
import pypdf
import docx
from bs4 import BeautifulSoup


def load_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def load_html(path: str) -> str:
    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")
    return soup.get_text(separator="\n", strip=True)


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


LOADERS = {
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm":  load_html,
    ".txt":  load_text,
    ".md":   load_text,
}


def load_document(path: str) -> dict:
    ext = Path(path).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    text = LOADERS[ext](path)
    return {
        "doc_id": Path(path).stem,
        "source": path,
        "text":   text.strip(),
    }


def load_directory(folder: str) -> Iterator[dict]:
    for path in Path(folder).rglob("*"):
        if path.suffix.lower() in LOADERS:
            try:
                yield load_document(str(path))
            except Exception as e:
                print(f"Skipping {path}: {e}")


def load_from_beir(corpus) -> Iterator[dict]:
    """Convert a BeIR corpus dataset directly into loader format."""
    for doc in corpus:
        yield {
            "doc_id": doc["_id"],
            "source": "beir/fiqa",
            "text":   (doc["title"] + " " + doc["text"]).strip(),
        }


if __name__ == "__main__":
    from evaluation.beir_loader import load_fiqa
    corpus, _, _ = load_fiqa()

    docs = list(load_from_beir(corpus))
    print(f"Loaded {len(docs)} documents")
    print("Sample:", docs[0])