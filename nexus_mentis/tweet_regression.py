"""Diagnóstico reproducible de diez tweets de Twitter/X; no es un benchmark.

No guarda textos originales ni sus traducciones en el reporte, solo identificadores
y métricas. Separa neutral y mixto de la evaluación binaria.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from nexus_mentis.bilingual_eval import _sha
from nexus_mentis.linguistic_signals import inspect_signals
from nexus_mentis.model import classify, load_artifacts
from nexus_mentis.translation import (
    ArgosSpanishEnglish, TranslationFailureError, TranslationUnavailableError, Translator
)
from nexus_mentis.translation_quality import guarded_translate, TranslationFidelityError

LOGGER = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = PROJECT / "evaluation" / "casos_tweets_v091.json"
LABELS = ("negativo", "positivo")
GROUPS = {"positivo_literal", "negativo_literal", "neutral", "sarcasmo", "negaciones", "mixto"}


def load_tweet_cases(path: Path) -> list[dict[str, Any]]:
    """Valida estructura, tipos, etiquetas, tamaños y unicidad de IDs."""
    source = path.expanduser().resolve()
    if not source.is_file() or source.suffix != ".json" or source.stat().st_size > 100_000:
        raise ValueError("Archivo JSON de diagnóstico no encontrado o demasiado grande.")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("JSON de pruebas inválido.") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Esquema de diagnóstico no soportado.")
    rows = data.get("cases")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("El conjunto debe contener de 1 a 100 casos.")
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Caso de diagnóstico no válido.")
        case_id = row.get("id")
        if (not isinstance(case_id, str) or not 1 <= len(case_id) <= 40
                or not case_id.isascii() or not case_id.isalnum()
                or case_id in ids):
            raise ValueError("Identificador de caso duplicado o inválido.")
        ids.add(case_id)
        if row.get("language") not in ("es", "en") or row.get("group") not in GROUPS:
            raise ValueError("Idioma o grupo de prueba inválido.")
        label = row.get("expected")
        if label not in (*LABELS, None):
            raise ValueError("Etiqueta fuera del modelo binario.")
        if (row["group"] in ("neutral", "mixto")) != (label is None):
            raise ValueError("Casos neutrales o mixtos deben estar fuera del F1 binario.")
        text = row.get("text")
        if not isinstance(text, str) or not 3 <= len(text.strip()) <= 1000:
            raise ValueError("Texto de prueba fuera del rango permitido.")
    return rows


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "mean_ms": round(statistics.mean(ordered), 3),
        "p95_ms": round(ordered[index], 3),
    }


def evaluate_tweets(
    rows: list[dict[str, Any]],
    translator: Translator,
    predictor: Callable[[str], str],
) -> dict[str, Any]:
    """Ejecuta el mismo pipeline de la API sin red ni archivo de registro."""
    results: list[dict[str, Any]] = []
    expected: list[str] = []
    predicted: list[str] = []
    group_stats: dict[str, dict[str, int]] = {}
    times: dict[str, list[float]] = {"es": [], "en": [], "translation": []}
    warnings: Counter[str] = Counter()
    blocked = 0
    for row in rows:
        group = row["group"]
        summary = group_stats.setdefault(group, {
            "total": 0, "eligible": 0, "classified": 0, "correct": 0, "blocked": 0
        })
        summary["total"] += 1
        label = row["expected"]
        if label is not None:
            summary["eligible"] += 1
        text = row["text"]
        started = time.perf_counter_ns()
        translation_ms = 0.0
        flags: list[str] = []
        predicted_label: str | None = None
        status = "classified"
        try:
            if row["language"] == "es":
                translation_start = time.perf_counter_ns()
                if not translator.is_available():
                    raise TranslationUnavailableError("Paquete lingüístico no disponible.")
                checked = guarded_translate(text, translator)
                assessed = inspect_signals(text, checked.text, "es")
                translation_ms = (time.perf_counter_ns() - translation_start) / 1_000_000
                flags = list(assessed.flags)
                if checked.paraphrase_used:
                    flags.append("controlled_paraphrase")
                model_input = assessed.translated_text
                times["translation"].append(translation_ms)
            else:
                assessed = inspect_signals(text, text, "en")
                model_input = assessed.translated_text
                flags = list(assessed.flags)
            predicted_label = predictor(model_input)
            if predicted_label not in LABELS:
                raise ValueError("El modelo devolvió una clase fuera del vocabulario.")
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            times[row["language"]].append(elapsed_ms)
            if label is not None:
                summary["classified"] += 1
                expected.append(label)
                predicted.append(predicted_label)
                summary["correct"] += int(predicted_label == label)
            warnings.update(flags)
        except (TranslationFidelityError, TranslationFailureError, TranslationUnavailableError) as error:
            status = "blocked"
            blocked += 1
            summary["blocked"] += 1
            LOGGER.warning("Tweet diagnóstico %s bloqueado: %s", row["id"], type(error).__name__)
        results.append({
            "id": row["id"], "group": group, "language": row["language"],
            "expected": label, "predicted": predicted_label,
            "status": status, "flags": flags,
            "total_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
            "translation_ms": round(translation_ms, 3),
        })
    scores = None
    if expected:
        prec, rec, f1, support = precision_recall_fscore_support(
            expected, predicted, labels=list(LABELS), zero_division=0
        )
        scores = {
            "n": len(expected),
            "accuracy": round(float(accuracy_score(expected, predicted)), 6),
            "precision_macro": round(float(prec.mean()), 6),
            "recall_macro": round(float(rec.mean()), 6),
            "f1_macro": round(float(f1.mean()), 6),
            "class_support": dict(zip(LABELS, (int(x) for x in support))),
        }
    return {
        "total_cases": len(rows),
        "eligible_binary_cases": sum(r["expected"] is not None for r in rows),
        "excluded_nonbinary_cases": sum(r["expected"] is None for r in rows),
        "classified_cases": len(rows) - blocked,
        "blocked_cases": blocked,
        "coverage": round((len(rows) - blocked) / len(rows), 6),
        "binary_scores_on_classified": scores,
        "warning_counts": dict(warnings),
        "by_group": group_stats,
        "timings": {
            "spanish_pipeline": _summary(times["es"]),
            "english_pipeline": _summary(times["en"]),
            "translation": _summary(times["translation"]),
            "scope": "Ejecución local secuencial sin HTTP, navegador ni red. Incluye inicialización.",
        },
        "cases": results,
    }


def write_tweet_report(
    results: dict[str, Any], cases_path: Path, model_report_path: Path, output_dir: Path,
    *, replace: bool = False,
) -> Path:
    """Persistencia atómica con respaldo versionado y sin guardar texto de usuarios."""
    folder = output_dir.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "report.json"
    if output.exists():
        if not replace:
            raise FileExistsError("El reporte ya existe; usa --replace para respaldarlo.")
        backup = folder / f"report_backup_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
        output.replace(backup)
    document = {
        "schema_version": 1,
        "version": "v0.9.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_report_sha256": _sha(model_report_path),
        "cases_dataset_sha256": _sha(cases_path),
        "limitations": (
            "Diez casos redactados para detectar fallos; no son una muestra aleatoria "
            "ni una evaluación general de la precisión en español. "
            "Neutral y mixto están excluidos de las métricas binarias. "
            "Las alertas no modifican las etiquetas."
        ),
        **results,
    }
    temp = output.with_suffix(".json.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regresión de 10 tweets con modelos reales.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    model_dir = args.model.expanduser().resolve()
    report = model_dir / "report.json"
    if not report.is_file():
        parser.error("Falta report.json del modelo entrenado.")
    model, metadata = load_artifacts(model_dir)
    if metadata.get("demo"):
        parser.error("Se requiere el modelo real, no el de demostración.")
    rows = load_tweet_cases(args.cases)
    translator = ArgosSpanishEnglish()
    if any(row["language"] == "es" for row in rows) and not translator.is_available():
        parser.error("Falta el paquete local ES→EN.")
    results = evaluate_tweets(rows, translator, lambda s: classify(model, s))
    output = write_tweet_report(
        results, args.cases, report, args.out, replace=args.replace
    )
    print(json.dumps({
        "status": "completed", "report": str(output),
        "eligible_binary": results["eligible_binary_cases"],
        "classified_binary": results["binary_scores_on_classified"]["n"]
        if results["binary_scores_on_classified"] else 0,
        "errors": sum(
            r["expected"] is not None and r["predicted"] is not None
            and r["predicted"] != r["expected"] for r in results["cases"]
        ),
        "blocked": results["blocked_cases"],
        "diagnostic_f1_macro": (
            results["binary_scores_on_classified"]["f1_macro"]
            if results["binary_scores_on_classified"] else None
        ),
        "scope": "Diagnóstico curado, no precisión generalizable.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
