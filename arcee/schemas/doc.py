from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Doc:
    doc_name: str
    doc_text: str
    meta: Optional[Dict] = None

    def dict(self) -> Dict[str, str]:
        return {"doc_name": self.doc_name, "doc_text": self.doc_text, **(self.meta or {})}
