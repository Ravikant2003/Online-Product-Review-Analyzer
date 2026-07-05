from __future__ import annotations

import csv
import io
from collections.abc import Iterable


TEXT_COLUMN_ALIASES = (
    "Tweet Text",
    "review",
    "reviews",
    "text",
    "comment",
    "comments",
    "content",
    "message",
    "body",
)


def normalize_header(header: object) -> str:
    return " ".join(str(header or "").replace("_", " ").replace("-", " ").split()).lower()


def find_text_column(headers: Iterable[object]) -> str | None:
    normalized_aliases = {normalize_header(alias) for alias in TEXT_COLUMN_ALIASES}
    for header in headers:
        if normalize_header(header) in normalized_aliases:
            return str(header)
    return None


def read_texts_from_csv(csv_input: bytes | str, limit: int = 50) -> tuple[str, list[str]]:
    csv_text = csv_input.decode("utf-8-sig") if isinstance(csv_input, bytes) else csv_input
    reader = csv.DictReader(io.StringIO(csv_text))

    if reader.fieldnames is None:
        raise ValueError("CSV file must include a header row.")

    column = find_text_column(reader.fieldnames)
    if column is None:
        aliases = ", ".join(TEXT_COLUMN_ALIASES)
        raise ValueError(f"CSV must include one text column such as: {aliases}.")

    texts: list[str] = []
    for row in reader:
        value = row.get(column)
        cleaned = " ".join(str(value or "").split())
        if cleaned:
            texts.append(cleaned)
        if len(texts) >= limit:
            break

    if not texts:
        raise ValueError("CSV has no non-empty text values.")

    return column, texts
