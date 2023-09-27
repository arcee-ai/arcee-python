from dataclasses import asdict, dataclass


@dataclass
class Doc:
    doc_name: str
    doc_text: str
    meta: dict[str, str | int | float] | None = None

    def dict(self) -> dict[str, str]:
        return asdict(self)
