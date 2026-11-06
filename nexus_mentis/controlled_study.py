"""v0.9.2: comparación pareada reproducible de representaciones TF-IDF.

No utiliza los diez tweets curados para seleccionar hiperparámetros.
Los dos competidores se entrenan de cero sobre particiones idénticas.
La selección utiliza SOLO la validación; la prueba reservada se abre una vez.
No modifica el modelo actualmente utilizado por la aplicación.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from nexus_mentis.data import Audit, load_sentiment140
from nexus_mentis.model import score_predictions
from nexus_mentis.source import describe_provenance

LOGGER = logging.getLogger(__name__)
NAMES = ("Palabras (1-2)", "Palabras + caracteres (3-5)")
LABELS = (0, 1)
MIN_VALIDATION_GAIN = 0.003  # Umbral fijado de antemano: 0,3 pp de F1 macro


def make_model(name: str, seed: int = 42) -> Pipeline:
    """Modelos de la misma familia con representación distinta y semilla fija."""
    word = TfidfVectorizer(
        min_df=2, max_df=0.995, max_features=40_000,
        ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
    )
    if name == NAMES[0]:
        features: Any = word
    elif name == NAMES[1]:
        characters = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=3, max_features=45_000, sublinear_tf=True,
            strip_accents="unicode", dtype=np.float32,
        )
        features = FeatureUnion([("words", word), ("chars", characters)])
    else:
        raise ValueError("Representación desconocida.")
    return Pipeline([
        ("tfidf", features),
        ("classifier", LogisticRegression(
            C=1.0, max_iter=500, random_state=seed
        )),
    ])


def validate_frame(frame: pd.DataFrame) -> None:
    """Impide entrenamiento con entradas vacías, etiquetas inválidas o duplicados."""
    if not isinstance(frame, pd.DataFrame) or not {"text", "label"} <= set(frame):
        raise ValueError("Se requieren columnas text y label.")
    if frame.empty or frame["text"].isna().any() or frame["label"].isna().any():
        raise ValueError("El dataset contiene valores ausentes o está vacío.")
    if not frame["label"].isin(LABELS).all():
        raise ValueError("Las etiquetas deben ser binarias (0/1).")
    if not frame["text"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Todos los textos deben ser cadenas no vacías.")
    if frame["text"].duplicated().any():
        raise ValueError("Hay textos duplicados: podrían causar fuga de datos.")
    if min(frame["label"].value_counts().reindex(LABELS, fill_value=0)) < 15:
        raise ValueError("No hay suficientes ejemplos por clase.")


def _digest_texts(frame: pd.DataFrame) -> str:
    """Huella agregada del conjunto sin almacenar tweets ni sus hashes individuales."""
    digest = hashlib.sha256()
    for item in sorted(frame["text"].tolist()):
        payload = item.encode("utf-8")
        digest.update(len(payload).to_bytes(4, "big"))
        digest.update(payload)
    return digest.hexdigest()


def split_fixed(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 estratificado, una sola vez antes de construir pipelines."""
    validate_frame(frame)
    dev, test = train_test_split(
        frame, test_size=0.15, stratify=frame["label"], random_state=seed
    )
    train, validation = train_test_split(
        dev, test_size=0.15 / 0.85, stratify=dev["label"], random_state=seed
    )
    sets = [set(part["text"]) for part in (train, validation, test)]
    if any(sets[i].intersection(sets[j]) for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Hay fuga de textos entre conjuntos.")
    return train, validation, test


def choose_candidate(validation_scores: dict[str, dict[str, float]],
                     min_gain: float = MIN_VALIDATION_GAIN) -> str:
    """Decisión pre-registrada. Ni un valor de prueba influye en la selección."""
    if not 0 <= min_gain < 1:
        raise ValueError("min_gain fuera de rango.")
    baseline, candidate = NAMES
    gain = validation_scores[candidate]["f1_macro"] - validation_scores[baseline]["f1_macro"]
    return candidate if gain >= min_gain else baseline


def paired_bootstrap_delta(
    truth: np.ndarray, baseline_pred: np.ndarray, candidate_pred: np.ndarray,
    seed: int = 42, repeats: int = 300,
) -> dict[str, float | int]:
    """IC percentil 95% de Δ F1 usando remuestreo PAREADO por índices.

    Intervalo descriptivo de este conjunto de prueba; no prueba causalidad.
    """
    from sklearn.metrics import f1_score
    if not 100 <= repeats <= 2_000:
        raise ValueError("repeats debe estar entre 100 y 2000.")
    actual, first, second = map(np.asarray, (truth, baseline_pred, candidate_pred))
    if not len(actual) or len(actual) != len(first) or len(first) != len(second):
        raise ValueError("Los vectores pareados deben tener la misma longitud no vacía.")
    if any(not np.isin(vec, LABELS).all() for vec in (actual, first, second)):
        raise ValueError("Las clases pareadas deben ser 0/1.")
    rng = np.random.default_rng(seed + 1_092)
    differences = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        indexes = rng.integers(0, len(actual), size=len(actual))
        differences[i] = (
            f1_score(actual[indexes], second[indexes], average="macro", zero_division=0)
            - f1_score(actual[indexes], first[indexes], average="macro", zero_division=0)
        )
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {
        "lower_pp": round(float(lower * 100), 4),
        "upper_pp": round(float(upper * 100), 4),
        "bootstrap_repeats": repeats,
        "note": "IC percentil 95% descriptivo sobre la prueba fija, no sobre tweets en español.",
    }


def _comparison_test(y: pd.Series, predictions: dict[str, np.ndarray],
                     seed: int, repeats: int) -> dict[str, Any]:
    baseline, candidate = NAMES
    truth = y.to_numpy()
    a = predictions[baseline]
    b = predictions[candidate]
    baseline_correct = (a == truth)
    candidate_correct = (b == truth)
    b_only = int(np.count_nonzero(~baseline_correct & candidate_correct))
    a_only = int(np.count_nonzero(baseline_correct & ~candidate_correct))
    disagreements = a_only + b_only
    exact_p = (
        float(binomtest(min(a_only, b_only), n=disagreements, p=0.5).pvalue)
        if disagreements else 1.0
    )
    score_a = score_predictions(y, a)
    score_b = score_predictions(y, b)
    return {
        "scores": {baseline: score_a, candidate: score_b},
        "confusion_matrices": {
            baseline: confusion_matrix(truth, a, labels=list(LABELS)).tolist(),
            candidate: confusion_matrix(truth, b, labels=list(LABELS)).tolist(),
        },
        "f1_delta_candidate_minus_baseline_pp": round(
            100 * (score_b["f1_macro"] - score_a["f1_macro"]), 4
        ),
        "accuracy_delta_candidate_minus_baseline_pp": round(
            100 * (score_b["accuracy"] - score_a["accuracy"]), 4
        ),
        "paired_counts": {
            "baseline_only_correct": a_only,
            "candidate_only_correct": b_only,
            "both_correct": int(np.count_nonzero(baseline_correct & candidate_correct)),
            "both_wrong": int(np.count_nonzero(~baseline_correct & ~candidate_correct)),
        },
        "mcnemar_exact_p": round(exact_p, 8),
        "f1_delta_bootstrap_ci95": paired_bootstrap_delta(truth, a, b, seed, repeats),
        "disclaimer": (
            "Comparación pareada con prueba idéntica. La selección ya se hizo por "
            "validación; la prueba no determina el ganador. Los valores p/IC son "
            "exploratorios y no acreditan rendimiento en español o sarcasmo."
        ),
    }


def run_study(
    frame: pd.DataFrame, *, seed: int = 42,
    bootstrap_repeats: int = 300,
) -> tuple[dict[str, Any], Pipeline]:
    """Entrena ambos competidores, selecciona en validación y evalúa en test."""
    if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Semilla no válida.")
    train, validation, test = split_fixed(frame, seed)
    LOGGER.info("Particiones fijas: train=%d, val=%d, test=%d",
                len(train), len(validation), len(test))
    validation_scores: dict[str, dict[str, float]] = {}
    times: dict[str, dict[str, float]] = {}
    for name in NAMES:
        model = make_model(name, seed)
        start = time.perf_counter()
        model.fit(train["text"], train["label"])
        train_seconds = time.perf_counter() - start
        predicted = model.predict(validation["text"])
        validation_scores[name] = score_predictions(validation["label"], predicted)
        times[name] = {"train_seconds": round(train_seconds, 3)}
        LOGGER.info("Validación %s F1=%.4f, tiempo=%.2fs",
                    name, validation_scores[name]["f1_macro"], train_seconds)
    chosen = choose_candidate(validation_scores)
    LOGGER.info("Selección por F1 de validación: %s", chosen)
    # Solo ahora se refita con train+val. El test continúa intacto.
    dev = pd.concat([train, validation], ignore_index=True)
    test_predictions: dict[str, np.ndarray] = {}
    selected_pipeline: Pipeline | None = None
    for name in NAMES:
        model = make_model(name, seed)
        start = time.perf_counter()
        model.fit(dev["text"], dev["label"])
        times[name]["refit_seconds"] = round(time.perf_counter() - start, 3)
        start = time.perf_counter()
        pred = model.predict(test["text"])
        times[name]["test_batch_seconds"] = round(time.perf_counter() - start, 3)
        test_predictions[name] = np.asarray(pred, dtype=int)
        if name == chosen:
            selected_pipeline = model
    if selected_pipeline is None:
        raise RuntimeError("No se seleccionó un pipeline.")
    result = {
        "schema_version": 1,
        "version": "v0.9.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sample_clean_rows": len(frame),
        "split": {
            "train": len(train), "validation": len(validation), "test": len(test),
            "train_sha256": _digest_texts(train),
            "validation_sha256": _digest_texts(validation),
            "test_sha256": _digest_texts(test),
        },
        "models": list(NAMES),
        "selection": {
            "selected": chosen,
            "metric": "f1_macro",
            "minimum_candidate_gain_pp": MIN_VALIDATION_GAIN * 100,
            "validation_f1_delta_candidate_minus_baseline_pp": round(
                100 * (validation_scores[NAMES[1]]["f1_macro"]
                       - validation_scores[NAMES[0]]["f1_macro"]), 4
            ),
            "rules": (
                "Se elige candidato solo si mejora F1 macro de validación al menos "
                "0,3 puntos porcentuales; jamás se usa test ni los 10 tweets curados."
            ),
        },
        "validation": validation_scores,
        "test": _comparison_test(test["label"], test_predictions, seed, bootstrap_repeats),
        "timing": times,
        "science": {
            "scope": "Sentiment140 original en inglés; binario, supervisión débil.",
            "reference": "Ambos modelos entrenados de cero; no se evalúa el modelo guardado v0.9.1 en su propia prueba potencialmente vista.",
            "deployment": "No se activa automáticamente el modelo seleccionado; antes hay que revisar calidad bilingüe, licencias y latencia.",
            "diagnostic_tweets": "Los diez tweets curados están excluidos de entrenamiento, selección y cálculo del benchmark.",
            "historical_test": "Por usar muestreo/semilla 42, la partición puede coincidir con el experimento 200k previo; no es una segunda prueba históricamente inédita.",
        },
        "environment": {
            "python": sys.version.split()[0],
            "sklearn": __import__("sklearn").__version__,
            "platform": platform.system(),
        },
    }
    return result, selected_pipeline


def persist_study(report: dict[str, Any], selected_model: Pipeline, output: Path) -> Path:
    """Publica directorio completo de manera atómica. Nunca sustituye resultados previos."""
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError("La carpeta de v0.9.2 ya existe; se preservan sus artefactos.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nexus_v092_", dir=destination.parent) as temp:
        folder = Path(temp)
        report_file = folder / "report.json"
        model_file = folder / "selected_model.joblib"
        # No almacenar entradas ni traducciones individuales.
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                               encoding="utf-8")
        joblib.dump(selected_model, model_file)
        if not report_file.is_file() or not model_file.is_file():
            raise OSError("No se pudieron crear todos los artefactos.")
        folder.rename(destination)
    return destination / "report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estudio v0.9.2: misma prueba, palabras frente a palabras+caracteres.")
    parser.add_argument("--data", type=Path, required=True, help="CSV original de Sentiment140")
    parser.add_argument("--out", type=Path, default=Path("artifacts/controlled_study_v092"))
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--allow-unverified", action="store_true", help="Solo para pruebas, no para resultados académicos.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        if args.out.expanduser().resolve().exists():
            raise FileExistsError("Ya existe el experimento. No se sobrescribe.")
        if not 500 <= args.max_rows <= 1_600_000:
            raise ValueError("--max-rows debe estar entre 500 y 1600000.")
        started = time.perf_counter()
        frame, audit = load_sentiment140(args.data, max_rows=args.max_rows, seed=args.seed)
        provenance = describe_provenance(args.data, audit.source_sha256)
        if provenance.get("status") != "official_verified" and not args.allow_unverified:
            raise ValueError("Origen no verificado. Para un CSV de prueba usa --allow-unverified.")
        report, winner = run_study(frame, seed=args.seed,
                                   bootstrap_repeats=args.bootstrap_repeats)
        report["data"] = {
            "audit": audit.as_dict(), "provenance": provenance,
            "sampling": "Reservorio aleatorio estratificado, depurado antes de partir.",
        }
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        filename = persist_study(report, winner, args.out)
        summary = {
            "status": "completed", "report": str(filename),
            "selected_by_validation": report["selection"]["selected"],
            "validation_f1": report["validation"],
            "test_f1_delta_pp": report["test"]["f1_delta_candidate_minus_baseline_pp"],
            "test_ci95": report["test"]["f1_delta_bootstrap_ci95"],
            "note": "Modelo vigente no sustituido; resultados no representan español ni sarcasmo.",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        LOGGER.error("Estudio v0.9.2 no completado (%s): %s", type(exc).__name__, str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


def public_study_view(document: Any, active_report: dict[str, Any] | None) -> dict[str, Any]:
    """Lista blanca para API: solo indicadores agregados; nunca textos ni rutas."""
    if active_report is None:
        return {"status": "model_missing"}
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        return {"status": "invalid"}
    try:
        active_hash = active_report["source_verification"]["csv_sha256"]
        provenance = document["data"]["provenance"]
        if (active_report["source_verification"].get("status") != "official_verified"
                or provenance["status"] != "official_verified"):
            return {"status": "unverified"}
        if provenance["csv_sha256"] != active_hash:
            return {"status": "stale"}
        selection = document["selection"]
        if selection["selected"] not in NAMES:
            return {"status": "invalid"}
        metrics = document["test"]["scores"]
        validation = document["validation"]
        for name in NAMES:
            for results in (metrics[name], validation[name]):
                for field in ("f1_macro", "accuracy", "precision_macro", "recall_macro"):
                    value = results[field]
                    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                        return {"status": "invalid"}
        raw_ci = document["test"]["f1_delta_bootstrap_ci95"]
        delta = document["test"]["f1_delta_candidate_minus_baseline_pp"]
        if any(type(x) not in (float, int) or not math.isfinite(x)
               for x in (delta, raw_ci["lower_pp"], raw_ci["upper_pp"])):
            return {"status": "invalid"}
        chosen = selection["selected"]
        sample = int(document["sample_clean_rows"])
        split = document["split"]
        if not 500 <= sample <= 1_600_000:
            return {"status": "invalid"}
        if sum(int(split[k]) for k in ("train", "validation", "test")) != sample:
            return {"status": "invalid"}
        return {
            "status": "completed",
            "selection": chosen,
            "minimum_gain_pp": MIN_VALIDATION_GAIN * 100,
            "selected_model_active": False,
            "counts": {key: int(split[key]) for key in ("train", "validation", "test")},
            "validation": {
                name: {k: round(validation[name][k], 6) for k in ("f1_macro", "accuracy")}
                for name in NAMES
            },
            "test": {
                name: {k: round(metrics[name][k], 6) for k in ("f1_macro", "accuracy")}
                for name in NAMES
            },
            "delta_f1_pp": delta,
            "ci95_delta_pp": {
                "lower": raw_ci["lower_pp"], "upper": raw_ci["upper_pp"]
            },
            "mcnemar_exact_p": document["test"]["mcnemar_exact_p"],
            "paired_counts": {
                k: int(document["test"]["paired_counts"][k]) for k in (
                    "baseline_only_correct", "candidate_only_correct",
                    "both_correct", "both_wrong"
                )
            },
            "scope": (
                "Comparación pareada, únicamente tweets originales en inglés. "
                "Selección en validación; la partición puede coincidir con el estudio 200k previo. "
                "El resultado no sustituye el modelo activo ni certifica sarcasmo o ES→EN."
            ),
        }
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return {"status": "invalid"}
