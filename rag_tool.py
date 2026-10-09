from pathlib import Path
import re

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - fallback for minimal environments
    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

POLICY_FILE = Path(__file__).resolve().parent / "university_it_policies.txt"
PATTERN = re.compile(
    r"\[POLICY id=(doc-\d+) topic=([a-z_]+)\]\s*\n"
    r"(.*?)\n\[END POLICY\]",
    flags=re.DOTALL,
)


def load_policy_documents(path=POLICY_FILE):
    text = Path(path).read_text(encoding="utf-8")
    matches = list(PATTERN.finditer(text))
    if not matches:
        raise ValueError(f"No [POLICY] sections found in {path}")

    documents = [
        Document(
            page_content=match.group(3).strip(),
            metadata={
                "source": Path(path).name,
                "policy_id": match.group(1),
                "topic": match.group(2),
            },
        )
        for match in matches
    ]

    if len({doc.metadata["policy_id"] for doc in documents}) != len(documents):
        raise ValueError("Duplicate policy IDs in knowledge base")
    return documents


# TO LOAD THE DOCUMENTS, USE:
documents = load_policy_documents()