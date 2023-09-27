from dataclasses import dataclass


@dataclass
class Doc:
    doc_name: str
    doc_text: str
    meta: dict | None = None

    def dict(self) -> dict[str, str]:
        return {"doc_name": self.doc_name, "doc_text": self.doc_text, **(self.meta or {})}
