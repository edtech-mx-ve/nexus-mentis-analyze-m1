"""v0.9.3: entrenamiento especializado EXPLORATORIO, sin promover modelos.

Comparación de dos Regresiones Logísticas de palabras+caracteres:
  referencia = Sentiment140 train;
  especializada = Sentiment140 train + ejemplos sintéticos de entrenamiento.

La validación normal de Sentiment140 y la sintética se usan para la selección;
el test compartido y los casos históricos se consultan DESPUÉS y nunca entrenan.
Resultados históricos no constituyen una prueba fresca ni validación en español.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from nexus_mentis.controlled_study import (
    NAMES, _comparison_test, _digest_texts, make_model, split_fixed, validate_frame,
)
from nexus_mentis.data import load_sentiment140, normalize_tweet
from nexus_mentis.model import score_predictions
from nexus_mentis.source import describe_provenance

LOGGER = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = PROJECT / "evaluation" / "specialized_cases_v093.json"
KNOWN_CASES = PROJECT / "evaluation" / "casos_tweets_v091.json"
REFERENCE = "Referencia: palabras+caracteres"
SPECIALIZED = "Especializado: palabras+caracteres+ejemplos"
LABELS = {0: "negativo", 1: "positivo"}
SYNTHETIC_WEIGHT = 40.0
MAX_MAIN_VALIDATION_DROP = 0.002  # 0.20 puntos porcentuales
MIN_CURATED_VALIDATION_GAIN = 0.04  # 4 puntos porcentuales
GROUPS = {
    "sarcasm_negative": 0,
    "contrast_positive": 1,
    "contrast_negative": 0,
    "literal_positive": 1,
    "literal_negative": 0,
}


def load_curated_cases(path: Path) -> dict[str, pd.DataFrame]:
    """Rechaza esquemas, fugas, etiquetas incoherentes o tamaños inesperados."""
    file = Path(path).expanduser().resolve()
    if not file.is_file() or file.suffix.lower() != ".json" or file.stat().st_size > 200_000:
        raise ValueError("Archivo de casos sintéticos ausente o excesivamente grande.")
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("No se pudo leer el JSON de ejemplos.") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Esquema de casos no compatible.")
    records = payload.get("examples")
    if not isinstance(records, list) or not 30 <= len(records) <= 300:
        raise ValueError("Se requieren entre 30 y 300 ejemplos.")
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    partitions: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": []
    }
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Caso sintético inválido.")
        ident, split, group, label, text = (
            item.get("id"), item.get("split"), item.get("group"),
            item.get("label"), item.get("text")
        )
        if (not isinstance(ident, str) or not ident.isascii()
                or not ident.isalnum() or len(ident) > 32 or ident in seen_ids):
            raise ValueError("ID de ejemplo inválido o duplicado.")
        if split not in partitions or group not in GROUPS or type(label) is not int:
            raise ValueError("Grupo, división o etiqueta inválida.")
        if GROUPS[group] != label:
            raise ValueError("Inconsistencia entre el grupo y la etiqueta declarada.")
        if not isinstance(text, str) or not 8 <= len(text.strip()) <= 300:
            raise ValueError("Texto de ejemplo inválido.")
        cleaned = normalize_tweet(text)
        if cleaned in seen_texts:
            raise ValueError("Duplicación de texto entre conjuntos sintéticos.")
        seen_texts.add(cleaned)
        seen_ids.add(ident)
        partitions[split].append({
            "id": ident, "text": cleaned, "label": label, "group": group,
        })
    result = {key: pd.DataFrame(rows) for key, rows in partitions.items()}
    for split, frame in result.items():
        if len(frame) < 5 or set(frame["label"]) != {0, 1}:
            raise ValueError(f"Partición sintética {split} insuficiente.")
    return result


def verify_separation(original_sets: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                      curated: dict[str, pd.DataFrame]) -> None:
    """Impide fugas exactas desde el corpus o hacia los diagnósticos conocidos."""
    train, validation, test = original_sets
    all_original = set(pd.concat([train["text"], validation["text"], test["text"]]))
    all_synthetic = set(pd.concat([part["text"] for part in curated.values()]))
    if all_original.intersection(all_synthetic):
        raise ValueError("Ejemplos sintéticos coinciden con Sentiment140.")
    known_english_inputs = [
        "What an excellent service! I just had to wait three hours to be served. 🙄",
        "I didn't expect much from this movie, but it ended up being fantastic.",
    ]
    if {normalize_tweet(x) for x in known_english_inputs}.intersection(all_synthetic):
        raise ValueError("Los dos tweets históricos no deben usarse para entrenar/elegir.")


def fit_specialized(base: pd.DataFrame, authored_train: pd.DataFrame,
                    seed: int = 42, weight: float = SYNTHETIC_WEIGHT) -> Pipeline:
    """Ajusta TF-IDF exclusivamente con entrenamiento + ejemplos nuevos.

    Pesos elegidos antes de las evaluaciones, sin optimizarlos con los dos tweets.
    """
    if not math.isfinite(weight) or weight <= 0 or weight > 100:
        raise ValueError("Peso de ejemplos sintéticos fuera de rango.")
    frame = pd.concat(
        [base[["text", "label"]], authored_train[["text", "label"]]],
        ignore_index=True
    )
    model = make_model("Palabras + caracteres (3-5)", seed)
    weights = np.concatenate((
        np.ones(len(base), dtype=np.float64),
        np.full(len(authored_train), weight, dtype=np.float64),
    ))
    model.fit(frame["text"], frame["label"], classifier__sample_weight=weights)
    return model


def _evaluate_curated(
    frame: pd.DataFrame, baseline: Pipeline, candidate: Pipeline,
) -> dict[str, Any]:
    """Evalúa textos redactados y devuelve SOLO métricas y grupos agregados."""
    original = np.asarray(baseline.predict(frame["text"]), dtype=int)
    alternative = np.asarray(candidate.predict(frame["text"]), dtype=int)
    result: dict[str, Any] = {
        "n": len(frame),
        "reference": score_predictions(frame["label"], original),
        "specialized": score_predictions(frame["label"], alternative),
        "by_group": {},
    }
    for group, subset in frame.groupby("group", sort=True):
        positions = frame.index.get_indexer(subset.index)
        truth = subset["label"].to_numpy(dtype=int)
        result["by_group"][group] = {
            "n": len(subset),
            "reference_correct": int(np.sum(original[positions] == truth)),
            "specialized_correct": int(np.sum(alternative[positions] == truth)),
        }
    return result


def _choose(main_scores: dict[str, dict[str, float]],
            targeted_scores: dict[str, Any]) -> tuple[str, str]:
    """Selección registrada, sin observar los resultados de test ni diagnósticos."""
    delta_main = main_scores[SPECIALIZED]["f1_macro"] - main_scores[REFERENCE]["f1_macro"]
    delta_targeted = (
        targeted_scores["specialized"]["f1_macro"] -
        targeted_scores["reference"]["f1_macro"]
    )
    if delta_main >= -MAX_MAIN_VALIDATION_DROP and delta_targeted >= MIN_CURATED_VALIDATION_GAIN:
        return SPECIALIZED, "Mejora validación curada ≥4 pp sin perder más de 0,2 pp en validación Sentiment140."
    return REFERENCE, "No cumplió simultáneamente los umbrales de validación preestablecidos."


def _known_failures(baseline: Pipeline, candidate: Pipeline,
                    translator: Any | None = None) -> dict[str, Any]:
    """Prueba post hoc, nunca usada para selección. Una traducción por tweet."""
    if translator is None or not translator.is_available():
        return {"status": "not_run", "reason": "Paquete ES→EN no disponible."}
    from nexus_mentis.tweet_regression import load_tweet_cases
    from nexus_mentis.translation_quality import guarded_translate
    from nexus_mentis.linguistic_signals import inspect_signals
    cases = load_tweet_cases(KNOWN_CASES)
    requested = {"tweet04", "tweet05"}
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case["id"] not in requested:
            continue
        try:
            checked = guarded_translate(case["text"], translator)
            assessed = inspect_signals(case["text"], checked.text, "es")
        except Exception as exc:
            # Falla cerrado: ninguna predicción se produce usando un texto dudoso.
            from nexus_mentis.translation import TranslationFailureError, TranslationUnavailableError
            from nexus_mentis.translation_quality import TranslationFidelityError
            if not isinstance(exc, (TranslationFailureError, TranslationUnavailableError, TranslationFidelityError)):
                raise
            rows.append({"id": case["id"], "status": "blocked"})
            continue
        normal = normalize_tweet(assessed.translated_text)
        first = int(baseline.predict([normal])[0])
        second = int(candidate.predict([normal])[0])
        rows.append({
            "id": case["id"], "status": "classified", "expected": case["expected"],
            "reference": LABELS[first], "specialized": LABELS[second],
            "correct_reference": LABELS[first] == case["expected"],
            "correct_specialized": LABELS[second] == case["expected"],
            "flags": list(assessed.flags),
        })
    return {"status": "completed" if len(rows) == 2 else "incomplete", "cases": rows,
            "note": "Dos errores conocidos y revisados previamente; diagnóstico post hoc, NO prueba ciega."}


def run_study(
    frame: pd.DataFrame, curated: dict[str, pd.DataFrame],
    *, seed: int = 42, bootstrap_repeats: int = 200,
    translator: Any | None = None,
) -> tuple[dict[str, Any], Pipeline]:
    """Entrena, selecciona SOLO en validación y después compara sobre prueba fija."""
    if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Semilla inválida.")
    if not 100 <= bootstrap_repeats <= 2_000:
        raise ValueError("bootstrap_repeats fuera de rango.")
    validate_frame(frame)
    train, val, test = split_fixed(frame, seed)
    verify_separation((train, val, test), curated)
    timings: dict[str, float] = {}

    start = time.perf_counter()
    reference = make_model("Palabras + caracteres (3-5)", seed)
    reference.fit(train["text"], train["label"])
    timings["reference_train_seconds"] = round(time.perf_counter() - start, 3)

    start = time.perf_counter()
    specialist = fit_specialized(train, curated["train"], seed)
    timings["specialized_train_seconds"] = round(time.perf_counter() - start, 3)

    val_scores = {
        REFERENCE: score_predictions(val["label"], reference.predict(val["text"])),
        SPECIALIZED: score_predictions(val["label"], specialist.predict(val["text"])),
    }
    validation_curated = _evaluate_curated(curated["validation"], reference, specialist)
    selected, reason = _choose(val_scores, validation_curated)

    # NO se incorporan ejemplos sintéticos de validación ni test al reentrenar.
    dev = pd.concat([train, val], ignore_index=True)
    start = time.perf_counter()
    reference_final = make_model("Palabras + caracteres (3-5)", seed)
    reference_final.fit(dev["text"], dev["label"])
    timings["reference_refit_seconds"] = round(time.perf_counter() - start, 3)

    start = time.perf_counter()
    specialist_final = fit_specialized(dev, curated["train"], seed)
    timings["specialized_refit_seconds"] = round(time.perf_counter() - start, 3)

    test_predictions = {
        REFERENCE: np.asarray(reference_final.predict(test["text"]), dtype=int),
        SPECIALIZED: np.asarray(specialist_final.predict(test["text"]), dtype=int),
    }
    # Reutilizar el evaluador pareado validado, adaptando nombres sin alterar predicciones.
    raw_comparison = _comparison_test(
        test["label"],
        {NAMES[0]: test_predictions[REFERENCE],
         NAMES[1]: test_predictions[SPECIALIZED]},
        seed, bootstrap_repeats
    )
    comparison = {
        **raw_comparison,
        "scores": {
            REFERENCE: raw_comparison["scores"][NAMES[0]],
            SPECIALIZED: raw_comparison["scores"][NAMES[1]],
        },
        "confusion_matrices": {
            REFERENCE: raw_comparison["confusion_matrices"][NAMES[0]],
            SPECIALIZED: raw_comparison["confusion_matrices"][NAMES[1]],
        },
    }
    curated_test = _evaluate_curated(curated["test"], reference_final, specialist_final)
    known = _known_failures(reference_final, specialist_final, translator)

    document = {
        "schema_version": 1,
        "version": "v0.9.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sample_clean_rows": len(frame),
        "split": {
            "train": len(train), "validation": len(val), "test": len(test),
            "test_sha256": _digest_texts(test),
        },
        "curated": {
            "provenance": "100 textos ingleses sintéticos redactados para el proyecto; no tweets de Twitter/X.",
            "counts": {
                "train": len(curated["train"]),
                "validation": len(curated["validation"]),
                "test": len(curated["test"]),
            },
            "weight_train_only": SYNTHETIC_WEIGHT,
            "validation": validation_curated,
            "test": curated_test,
        },
        "selection": {
            "selected": selected,
            "reason": reason,
            "main_validation_max_loss_pp": 100 * MAX_MAIN_VALIDATION_DROP,
            "targeted_validation_min_gain_pp": 100 * MIN_CURATED_VALIDATION_GAIN,
            "rule": "Decisión tomada antes de consultar la prueba general y los 2 tweets conocidos.",
        },
        "main_validation": val_scores,
        "main_test": comparison,
        "known_failures": known,
        "timings": timings,
        "environment": {
            "python": sys.version.split()[0],
            "scikit_learn": __import__("sklearn").__version__,
            "platform": platform.system(),
        },
        "limitations": (
            "Estudio EXPLORATORIO: Sentiment140 contiene etiquetas débiles en inglés; "
            "se ha consultado antes su partición de prueba en v0.9.2. Los 100 ejemplos "
            "nuevos son sintéticos y fueron diseñados después de conocer fallos; "
            "el pequeño conjunto curado no permite generalización a Twitter/X. "
            "Los 2 tweets ya eran fallos conocidos: su análisis es post hoc. "
            "No se modifica el modelo activo, no hay despliegue automático."
        ),
    }
    # Un único modelo se guarda: el candidato especializado, incluso si no es seleccionado.
    return document, specialist_final


def persist(report: dict[str, Any], candidate: Pipeline, destination: Path) -> Path:
    """Publica un directorio completo sin sobrescribir ningún artefacto previo."""
    final = Path(destination).expanduser().resolve()
    if final.exists():
        raise FileExistsError("El estudio v0.9.3 ya existe; no se sobrescribe.")
    final.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nexus_v093_", dir=final.parent) as tmp:
        folder = Path(tmp)
        report_file = folder / "report.json"
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8"
        )
        joblib.dump(candidate, folder / "specialized_model.joblib")
        folder.rename(final)
    return final / "report.json"


def public_view(document: Any, active_report: dict[str, Any] | None) -> dict[str, Any]:
    """Vista pública de indicadores, sin textos sintéticos, traducciones ni rutas."""
    if active_report is None:
        return {"status": "model_missing"}
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return {"status": "invalid"}
    try:
        if document["data"]["provenance"]["status"] != "official_verified":
            return {"status": "unverified"}
        active_sha = active_report["source_verification"]["csv_sha256"]
        if document["data"]["provenance"]["csv_sha256"] != active_sha:
            return {"status": "stale"}
        if document["selection"]["selected"] not in (REFERENCE, SPECIALIZED):
            return {"status": "invalid"}
        scores = document["main_test"]["scores"]
        for name in (REFERENCE, SPECIALIZED):
            for metric in ("accuracy", "f1_macro"):
                if not 0 <= float(scores[name][metric]) <= 1:
                    return {"status": "invalid"}
        known = document.get("known_failures", {})
        safe_known = [
            {k: row.get(k) for k in ("id", "status", "expected", "reference",
                                      "specialized", "correct_reference", "correct_specialized")}
            for row in known.get("cases", []) if isinstance(row, dict)
            and row.get("id") in {"tweet04", "tweet05"}
        ]
        test = document["curated"]["test"]
        # No se emiten los textos de los datasets ni de los tweets problemáticos.
        return {
            "status": "completed",
            "selected_by_validation": document["selection"]["selected"],
            "selection_reason": document["selection"]["reason"],
            "validation_f1": {name: round(document["main_validation"][name]["f1_macro"], 6)
                              for name in (REFERENCE, SPECIALIZED)},
            "main_test_f1": {name: round(scores[name]["f1_macro"], 6)
                             for name in (REFERENCE, SPECIALIZED)},
            "delta_main_test_pp": document["main_test"]["f1_delta_candidate_minus_baseline_pp"],
            "ci95_delta_pp": document["main_test"]["f1_delta_bootstrap_ci95"],
            "curated_test": {
                "n": test["n"], "f1_reference": test["reference"]["f1_macro"],
                "f1_specialized": test["specialized"]["f1_macro"],
                "by_group": test["by_group"],
            },
            "known_failures": {
                "status": known.get("status", "not_run"),
                "cases": safe_known,
            },
            "deployment": "NO activado. Estudio exploratorio con ejemplos sintéticos, no validación en español.",
        }
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return {"status": "invalid"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v0.9.3: entrenamiento especializado exploratorio.")
    parser.add_argument("--data", required=True, type=Path, help="Sentiment140 CSV original.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/specialized_study_v093"))
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--allow-unverified", action="store_true",
                        help="Solo para pruebas locales con CSV sintético.")
    parser.add_argument("--no-translate", action="store_true",
                        help="Omitir diagnóstico conocido de ES→EN.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        if args.out.expanduser().resolve().exists():
            raise FileExistsError("Ya existe el resultado; elige otra carpeta.")
        if not 500 <= args.max_rows <= 1_600_000:
            raise ValueError("max-rows fuera del rango [500, 1600000].")
        data, audit = load_sentiment140(args.data, max_rows=args.max_rows, seed=args.seed)
        provenance = describe_provenance(args.data, audit.source_sha256)
        if provenance.get("status") != "official_verified" and not args.allow_unverified:
            raise ValueError("Fuente no verificada. La prueba sintética requiere --allow-unverified.")
        curated = load_curated_cases(args.cases)
        translator = None
        if not args.no_translate:
            from nexus_mentis.translation import ArgosSpanishEnglish
            translator = ArgosSpanishEnglish()
        started = time.perf_counter()
        report, specialist = run_study(
            data, curated, seed=args.seed,
            bootstrap_repeats=args.bootstrap_repeats, translator=translator
        )
        report["data"] = {
            "audit": audit.as_dict(), "provenance": provenance,
            "curated_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        }
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        target = persist(report, specialist, args.out)
        public = {
            "status": "completed", "report": str(target),
            "selected_by_validation": report["selection"]["selected"],
            "f1_sentiment140": {
                name: report["main_test"]["scores"][name]["f1_macro"]
                for name in (REFERENCE, SPECIALIZED)
            },
            "f1_curated": {
                "reference": report["curated"]["test"]["reference"]["f1_macro"],
                "specialized": report["curated"]["test"]["specialized"]["f1_macro"],
            },
            "known_failures": report["known_failures"],
            "note": "Diagnóstico exploratorio; no se activa ningún modelo.",
        }
        print(json.dumps(public, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        LOGGER.error("Estudio especializado falló (%s): %s", type(exc).__name__, str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
