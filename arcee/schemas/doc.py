from dataclasses import dataclass
from typing import Optional


@dataclass
class Doc:
    doc_name: str
    doc_text: str
    meta: Optional[dict] = None

    def dict(self) -> dict[str, str]:
        return {"doc_name": self.doc_name, "doc_text": self.doc_text, **(self.meta or {})}
