"""Lectura, auditoría y depuración reproducible de Sentiment140."""
from __future__ import annotations

import csv
import hashlib
import html
import logging
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@\w+")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Audit:
    """Contadores del origen y de la muestra depurada (ámbitos diferenciados)."""

    source_rows: int
    source_file_bytes: int
    source_sha256: str
    source_valid_rows: int
    malformed_rows: int
    invalid_label_rows: int
    empty_text_rows: int
    source_labels: dict[str, int]
    sampled_valid_rows: int
    sampled_conflicting_texts_removed: int
    sampled_conflicting_records_removed: int
    sampled_duplicates_removed: int
    clean_rows: int
    clean_labels: dict[str, int]
    sample_limit: int | None
    seed: int

    def as_dict(self) -> dict[str, Any]:
        """Devuelve resumen JSON-serializable."""
        return asdict(self)


def normalize_tweet(value: str) -> str:
    """Normaliza texto preservando negaciones, hashtags, emojis y expresiones."""
    if not isinstance(value, str):
        raise TypeError("El texto debe ser una cadena.")
    text = unicodedata.normalize("NFKC", html.unescape(value)).strip().lower()
    text = _URL.sub(" urltoken ", text)
    text = _MENTION.sub(" usermention ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def validate_path(path: Path) -> Path:
    """Comprueba ruta local, extensión y existencia del dataset."""
    if not isinstance(path, Path):
        raise TypeError("Se requiere una ruta pathlib.Path.")
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".csv":
        raise ValueError("El dataset debe ser un archivo .csv.")
    if not resolved.is_file():
        raise FileNotFoundError(f"No existe el dataset: {resolved}")
    if resolved.stat().st_size == 0:
        raise ValueError("El archivo CSV está vacío.")
    return resolved


def load_sentiment140(
    path: Path, max_rows: int | None = 50_000, seed: int = 42
) -> tuple[pd.DataFrame, Audit]:
    """Audita el archivo completo y obtiene una muestra aleatoria equilibrada.

    Formato original Sentiment140: target,id,date,flag,user,text; sin encabezado.
    Las etiquetas 0 y 4 se convierten respectivamente a 0 (negativo) y
    1 (positivo). Los conteos de duplicados se refieren a la muestra.
    """
    source = validate_path(path)
    if max_rows is not None and max_rows < 40:
        raise ValueError("max_rows debe ser >= 40, o None para todos los registros.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed debe ser un entero no negativo.")

    rng = random.Random(seed)
    quotas = {0: (max_rows or 0) // 2, 1: ((max_rows or 0) + 1) // 2}
    buckets: dict[int, list[tuple[str, int]]] = {0: [], 1: []}
    seen_eligible: Counter[int] = Counter()
    raw_stats: Counter[str] = Counter()
    raw_labels: Counter[str] = Counter()

    with source.open("r", encoding="latin-1", newline="") as stream:
        reader = csv.reader(stream)
        for row_number, row in enumerate(reader, start=1):
            if row_number == 1 and len(row) == 6 and row[0].strip().lower() == "target":
                continue
            raw_stats["source_rows"] += 1
            if len(row) != 6:
                raw_stats["malformed_rows"] += 1
                continue
            if row[0].strip() not in {"0", "4"}:
                raw_stats["invalid_label_rows"] += 1
                continue
            label = 0 if row[0].strip() == "0" else 1
            text = normalize_tweet(row[5])
            if not text:
                raw_stats["empty_text_rows"] += 1
                continue
            raw_stats["source_valid_rows"] += 1
            raw_labels["negativo" if label == 0 else "positivo"] += 1
            seen_eligible[label] += 1
            item = (text, label)
            if max_rows is None:
                buckets[label].append(item)
            else:
                bucket = buckets[label]
                capacity = quotas[label]
                if len(bucket) < capacity:
                    bucket.append(item)
                else:
                    position = rng.randrange(seen_eligible[label])
                    if position < capacity:
                        bucket[position] = item

    sampled = buckets[0] + buckets[1]
    if not sampled:
        raise ValueError("No hay tweets válidos con etiquetas 0/4.")
    frame = pd.DataFrame(sampled, columns=["text", "label"])
    sample_count = len(frame)

    by_text = frame.groupby("text", sort=False)["label"].nunique()
    conflicts = set(by_text[by_text > 1].index)
    conflict_rows = int(frame["text"].isin(conflicts).sum())
    if conflicts:
        frame = frame[~frame["text"].isin(conflicts)].copy()
    before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    duplicates = before_dedup - len(frame)
    clean_counts = frame["label"].value_counts().to_dict()
    if min(int(clean_counts.get(0, 0)), int(clean_counts.get(1, 0))) < 10:
        raise ValueError("Se necesitan al menos 10 tweets únicos de cada clase.")

    digest = hashlib.sha256()
    with source.open("rb") as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
    audit = Audit(
        source_file_bytes=source.stat().st_size,
        source_sha256=digest.hexdigest(),
        source_rows=int(raw_stats["source_rows"]),
        source_valid_rows=int(raw_stats["source_valid_rows"]),
        malformed_rows=int(raw_stats["malformed_rows"]),
        invalid_label_rows=int(raw_stats["invalid_label_rows"]),
        empty_text_rows=int(raw_stats["empty_text_rows"]),
        source_labels=dict(raw_labels),
        sampled_valid_rows=sample_count,
        sampled_conflicting_texts_removed=len(conflicts),
        sampled_conflicting_records_removed=conflict_rows,
        sampled_duplicates_removed=duplicates,
        clean_rows=len(frame),
        clean_labels={
            "negativo": int(clean_counts.get(0, 0)),
            "positivo": int(clean_counts.get(1, 0)),
        },
        sample_limit=max_rows,
        seed=seed,
    )
    LOGGER.info(
        "Auditoría: origen=%d válidos=%d muestra_limpia=%d",
        audit.source_rows,
        audit.source_valid_rows,
        audit.clean_rows,
    )
    return frame, audit
