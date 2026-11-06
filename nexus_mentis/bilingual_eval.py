"""Evaluación diagnóstica bilingüe ES→EN→TF-IDF→clasificador.

Las referencias son casos redactados para diagnóstico, no un benchmark
representativo ni datos independientes con anotación humana ciega.
Ejecutar requiere Argos instalado y un artefacto de entrenamiento real.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from nexus_mentis.linguistic_signals import inspect_signals
from nexus_mentis.model import classify, load_artifacts
from nexus_mentis.translation import (
    ArgosSpanishEnglish, TranslationFailureError, TranslationUnavailableError,
    Translator,
)
from nexus_mentis.translation_quality import TranslationFidelityError, guarded_translate

LOGGER = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = PROJECT / "evaluation" / "casos_bilingues_v08.jsonl"
_ALLOWED_GROUPS = {
    "positivo_literal", "negativo_literal", "negaciones", "sarcasmo",
    "mixto", "neutral",
}
_CLASSES = ["negativo", "positivo"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Carga y valida un conjunto acotado sin ejecutar contenido externo."""
    resolved = path.expanduser().resolve()
    if resolved.suffix != ".jsonl" or not resolved.is_file():
        raise ValueError("Se necesita un archivo JSONL de evaluación existente.")
    if resolved.stat().st_size > 1_000_000:
        raise ValueError("El archivo de casos supera el límite de 1 MB.")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSONL inválido en línea {line_no}.") from error
        if not isinstance(row, dict):
            raise ValueError(f"Caso inválido en línea {line_no}.")
        case_id = row.get("id")
        label = row.get("expected")
        group = row.get("group")
        if (not isinstance(case_id, str) or not case_id or len(case_id) > 40 or
                case_id in seen or group not in _ALLOWED_GROUPS or
                label not in ("negativo", "positivo", None)):
            raise ValueError(f"Metadatos de caso inválidos en línea {line_no}.")
        if (group == "neutral") != (label is None):
            raise ValueError(f"Referencia neutral inconsistente en línea {line_no}.")
        for field in ("es", "en_reference"):
            if not isinstance(row.get(field), str) or not 3 <= len(row[field]) <= 1000:
                raise ValueError(f"Texto de caso inválido en línea {line_no}.")
        seen.add(case_id)
        rows.append(row)
    if not rows or len(rows) > 500:
        raise ValueError("El conjunto debe contener entre 1 y 500 casos.")
    return rows


def _scores(expected: list[str], predicted: list[str]) -> dict[str, Any] | None:
    """Métricas solo para predicciones existentes y etiquetas binarias."""
    if not expected:
        return None
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=_CLASSES, zero_division=0
    )
    return {
        "n": len(expected),
        "accuracy": round(float(accuracy_score(expected, predicted)), 6),
        "precision_macro": round(float(precision.mean()), 6),
        "recall_macro": round(float(recall.mean()), 6),
        "f1_macro": round(float(f1.mean()), 6),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=_CLASSES).tolist(),
        "labels": _CLASSES,
        "class_support": dict(zip(_CLASSES, [int(x) for x in support])),
    }


def _timing(values: list[float]) -> dict[str, float] | None:
    """Latencias descriptivas en milisegundos; no incluye HTTP/red."""
    if not values:
        return None
    series = sorted(values)
    index = max(0, min(len(series) - 1, int(0.95 * (len(series) - 1) + 0.999999)))
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "p95_ms": round(series[index], 3),
    }


def evaluate_cases(
    rows: list[dict[str, Any]],
    translator: Translator,
    predictor: Callable[[str], str],
) -> dict[str, Any]:
    """Ejecuta las rutas inglés de referencia y español traducido.

    Permite inyección de traductor y predictor para pruebas sin grandes
    modelos. Los textos no se registran ni se incluyen en el resultado.
    """
    if not translator.is_available():
        raise TranslationUnavailableError("El paquete ES→EN no está disponible.")
    expected_es: list[str] = []
    predicted_es: list[str] = []
    predicted_en_paired: list[str] = []
    expected_en: list[str] = []
    predicted_en: list[str] = []
    rows_out: list[dict[str, Any]] = []
    times_es: list[float] = []
    times_en: list[float] = []
    translate_times: list[float] = []
    warnings: Counter[str] = Counter()
    group_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    blocked = 0
    disagreements = 0
    neutral_es: Counter[str] = Counter()

    for row in rows:
        en_start = time.perf_counter_ns()
        en_prediction = predictor(row["en_reference"])
        en_ms = (time.perf_counter_ns() - en_start) / 1_000_000
        if en_prediction not in _CLASSES:
            raise ValueError("El clasificador inglés devolvió una etiqueta no válida.")
        times_en.append(en_ms)
        started = time.perf_counter_ns()
        es_prediction: str | None = None
        signal_flags: list[str] = []
        message = "classified"
        translated_ms: float | None = None
        try:
            translation_start = time.perf_counter_ns()
            checked = guarded_translate(row["es"], translator)
            assessed = inspect_signals(row["es"], checked.text, "es")
            translated_ms = (time.perf_counter_ns() - translation_start) / 1_000_000
            es_prediction = predictor(assessed.translated_text)
            if es_prediction not in _CLASSES:
                raise ValueError("El clasificador español devolvió una etiqueta no válida.")
            signal_flags = list(assessed.flags)
            if checked.paraphrase_used:
                signal_flags.append("controlled_paraphrase")
            warnings.update(signal_flags)
            translate_times.append(translated_ms)
            times_es.append((time.perf_counter_ns() - started) / 1_000_000)
            if en_prediction != es_prediction:
                disagreements += 1
        except (TranslationFidelityError, TranslationFailureError,
                TranslationUnavailableError) as error:
            blocked += 1
            message = "blocked"
            # No registrar texto de terceros ni el mensaje de excepción.
            LOGGER.warning("Caso %s bloqueado por %s.", row["id"], type(error).__name__)

        record = {
            "id": row["id"], "group": row["group"], "expected": row["expected"],
            "english_reference_prediction": en_prediction,
            "translated_prediction": es_prediction,
            "status": message, "flags": signal_flags,
            "translation_ms": round(translated_ms, 3) if translated_ms is not None else None,
        }
        rows_out.append(record)
        group_records[row["group"]].append(record)
        if row["expected"] is None:
            if es_prediction is not None:
                neutral_es[es_prediction] += 1
        else:
            expected_en.append(row["expected"])
            predicted_en.append(en_prediction)
            if es_prediction is not None:
                expected_es.append(row["expected"])
                predicted_es.append(es_prediction)
                predicted_en_paired.append(en_prediction)

    breakdown = {}
    for group, entries in sorted(group_records.items()):
        scored = [r for r in entries if r["expected"] is not None]
        valid = [r for r in scored if r["translated_prediction"] is not None]
        correct = sum(r["translated_prediction"] == r["expected"] for r in valid)
        breakdown[group] = {
            "total": len(entries),
            "labelled": len(scored),
            "classified": len(valid),
            "correct": correct,
            "blocked": sum(r["status"] == "blocked" for r in entries),
            "accuracy_on_classified": round(correct / len(valid), 6) if valid else None,
            "warning_cases": sum(bool(r["flags"]) for r in entries),
        }

    paired_es = _scores(expected_es, predicted_es)
    paired_en = _scores(expected_es, predicted_en_paired)
    paired_delta_points = (
        round(100 * (paired_es["f1_macro"] - paired_en["f1_macro"]), 3)
        if paired_es is not None and paired_en is not None else None
    )
    return {
        "total_cases": len(rows),
        "binary_cases": sum(r["expected"] is not None for r in rows),
        "neutral_cases": sum(r["expected"] is None for r in rows),
        "classified_cases": len(rows) - blocked,
        "blocked_cases": blocked,
        "coverage": round((len(rows) - blocked) / len(rows), 6),
        "binary_coverage": round(
            len(expected_es) / sum(r["expected"] is not None for r in rows), 6
        ) if any(r["expected"] is not None for r in rows) else None,
        "translated_binary_scores": paired_es,
        "english_reference_binary_scores": _scores(expected_en, predicted_en),
        "english_reference_paired_binary_scores": paired_en,
        "paired_f1_delta_points_es_minus_en": paired_delta_points,
        "by_group": breakdown,
        "warning_counts": dict(warnings),
        "neutral_predictions_not_scored": dict(neutral_es),
        "translation_vs_english_disagreements": disagreements,
        "timings": {
            "spanish_pipeline": _timing(times_es),
            "english_reference_inference": _timing(times_en),
            "translation": _timing(translate_times),
            "scope": "Ejecución local secuencial, sin red HTTP ni navegador.",
        },
        "cases": rows_out,
    }


def write_report(
    results: dict[str, Any], *,
    cases_path: Path, model_report: Path, out_dir: Path,
    replace: bool = False,
) -> Path:
    """Persistencia segura: reporte versionado y respaldo si se reemplaza."""
    target = out_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    output = target / "report.json"
    if output.exists() and not replace:
        raise FileExistsError(
            "El reporte ya existe. Usa --replace para conservar una copia anterior."
        )
    if output.exists():
        backup = target / f"report_backup_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
        output.replace(backup)
    manifest = {
        "schema_version": 1,
        "version": "v0.9",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_report_sha256": _sha(model_report),
        "case_dataset_sha256": _sha(cases_path),
        "dataset_label": "Casos sintéticos redactados para diagnóstico, no representativos.",
        "english_reference": "Traducciones de referencia redactadas, no validación humana ciega.",
        "evaluation_limitations": [
            "Muestra pequeña y seleccionada; no estima precisión poblacional en español.",
            "Las etiquetas propuestas no proceden de anotadores independientes.",
            "La traducción puede preservar la polaridad y perder información semántica.",
            "Los neutrales no entran en F1 binario; el modelo no predice neutral.",
            "F1 español se calcula solo sobre casos clasificados; informar cobertura.",
            "Comparar ES y EN con las mismas filas clasificadas; no usar EN sobre todas si hubo bloqueos.",
        ],
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
        },
        **results,
    }
    temp = output.with_suffix(".json.tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico ES→EN con modelo local real.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replace", action="store_true", help="Respalda el reporte previo.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    model_path = args.model.expanduser().resolve()
    model_report = model_path / "report.json"
    if not model_report.is_file():
        parser.error("No se encuentra report.json del modelo.")
    model, report = load_artifacts(model_path)
    if report.get("demo"):
        parser.error("La evaluación debe ejecutarse sobre un modelo real, no demo.")
    cases = load_cases(args.cases)
    translator = ArgosSpanishEnglish()
    if not translator.is_available():
        parser.error("El traductor ES→EN no está listo. Prepara Argos antes.")
    results = evaluate_cases(cases, translator, lambda text: classify(model, text))
    output = write_report(
        results, cases_path=args.cases, model_report=model_report,
        out_dir=args.out, replace=args.replace,
    )
    score = results["translated_binary_scores"]
    print(json.dumps({
        "status": "completed", "report": str(output),
        "total_cases": results["total_cases"],
        "blocked": results["blocked_cases"],
        "coverage": results["coverage"],
        "f1_macro_es_diagnostic": score["f1_macro"] if score else None,
        "warning": "Diagnóstico pequeño y curado: no equivale a precisión general en español.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
