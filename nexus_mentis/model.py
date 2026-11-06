"""Entrenamiento, comparación y evaluación reproducible de clasificadores."""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nexus_mentis.data import Audit, load_sentiment140, normalize_tweet
from nexus_mentis.source import describe_provenance

LOGGER = logging.getLogger(__name__)
MODEL_FILE = "model.joblib"
REPORT_FILE = "report.json"


def build_pipeline(name: str, seed: int = 42) -> Pipeline:
    """Construye un pipeline TF-IDF + clasificador sin estado compartido."""
    classifiers: dict[str, Any] = {
        "SVM lineal": LinearSVC(C=1.0, random_state=seed, max_iter=4000),
        "Regresión logística": LogisticRegression(
            C=1.0, max_iter=500, random_state=seed
        ),
        "Naive Bayes": MultinomialNB(alpha=1.0),
    }
    if name not in classifiers:
        raise ValueError(f"Modelo desconocido: {name}")
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=2,
                    max_df=0.995,
                    max_features=40_000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("classifier", classifiers[name]),
        ]
    )


def score_predictions(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calcula métricas comparables, en particular F1 macro para clases."""
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "f1_macro": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(actual, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(actual, predicted, average="macro", zero_division=0)),
    }


def _save_charts(
    out_dir: Path,
    counts: dict[str, int],
    matrix: list[list[int]],
) -> None:
    """Guarda figuras no interactivas para el informe académico."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(["Negativo", "Positivo"], [counts["negativo"], counts["positivo"]])
    ax.set_ylabel("Tweets")
    ax.set_title("Distribución del conjunto depurado")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    chart = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(chart, ax=ax)
    ax.set_xticks([0, 1], ["Negativo", "Positivo"])
    ax.set_yticks([0, 1], ["Negativo", "Positivo"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión — prueba reservada")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="white" if matrix[i][j] > np.max(matrix) / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)



def benchmark_inference(model: Any, texts: pd.Series, seed: int = 42,
                        repeats: int = 100) -> dict[str, float | int | str]:
    """Mide inferencia individual local, excluyendo red, API y navegador.

    Los resultados dependen del hardware, carga del equipo y tamaño de entrada.
    Usa una muestra reservada (nunca participa en el ajuste de los modelos).
    No registra el contenido de ninguna publicación.
    """
    if len(texts) == 0 or repeats < 1:
        raise ValueError("Se necesita una muestra no vacía y repeats >= 1.")
    rng = np.random.default_rng(seed)
    sample = texts.iloc[rng.integers(0, len(texts), size=min(repeats, len(texts)))].tolist()
    model.predict([sample[0]])  # calentamiento fuera de la medición
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 ** 2)
    latencies_ms: list[float] = []
    for text in sample:
        start_ns = time.perf_counter_ns()
        model.predict([text])
        latencies_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)
    rss_after = process.memory_info().rss / (1024 ** 2)
    total_seconds = sum(latencies_ms) / 1000.0
    return {
        "scope": "inferencia individual local del pipeline; excluye HTTP y navegador",
        "sample_size": len(sample),
        "mean_ms": round(float(np.mean(latencies_ms)), 4),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 4),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "throughput_tweets_per_s": round(len(sample) / max(total_seconds, 1e-9), 2),
        "rss_before_mb": round(rss_before, 2),
        "rss_after_mb": round(rss_after, 2),
        "rss_delta_mb": round(rss_after - rss_before, 2),
    }


def fit_experiment(
    frame: pd.DataFrame,
    audit: Audit,
    output_dir: Path,
    seed: int = 42,
    cv_folds: int = 3,
    force: bool = False,
    demo: bool = False,
    source_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compara modelos y persiste el ganador sobre una prueba no vista.

    El split 70/15/15 es estratificado y se hace tras eliminar duplicados.
    Las transformaciones TF-IDF se ajustan dentro de cada pipeline.
    Los modelos se seleccionan exclusivamente por la validación.
    """
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Faltan columnas text y label.")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Existen etiquetas fuera de 0/1.")
    if frame["text"].isna().any() or frame["text"].str.strip().eq("").any():
        raise ValueError("Existen textos ausentes o vacíos.")
    if frame["text"].duplicated().any():
        raise ValueError("Los textos deben estar deduplicados antes del split.")
    if min(frame["label"].value_counts().to_dict().values()) < 10:
        raise ValueError("Insuficientes observaciones por clase.")
    if cv_folds not in {0, 2, 3, 4, 5}:
        raise ValueError("cv_folds debe ser 0 o un entero entre 2 y 5.")

    out = output_dir.expanduser().resolve()
    if out.exists() and not out.is_dir():
        raise ValueError("La ruta de salida debe ser un directorio.")
    files = [MODEL_FILE, REPORT_FILE, "class_distribution.png", "confusion_matrix.png"]
    if not force and any((out / name).exists() for name in files):
        raise FileExistsError("Los artefactos existen. Usa --force para sustituirlos.")
    out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    dev, test = train_test_split(
        frame, test_size=0.15, random_state=seed, stratify=frame["label"]
    )
    train, validation = train_test_split(
        dev,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=dev["label"],
    )
    LOGGER.info(
        "Particiones: entrenamiento=%d validación=%d prueba=%d",
        len(train), len(validation), len(test),
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(train), 1)), train["label"])
    baseline_metrics = score_predictions(
        validation["label"], baseline.predict(np.zeros((len(validation), 1)))
    )

    comparison: list[dict[str, Any]] = []
    for name in ["Naive Bayes", "Regresión logística", "SVM lineal"]:
        pipeline = build_pipeline(name, seed=seed)
        began = time.perf_counter()
        pipeline.fit(train["text"], train["label"])
        elapsed = time.perf_counter() - began
        scores = score_predictions(
            validation["label"], pipeline.predict(validation["text"])
        )
        comparison.append(
            {"model": name, "train_seconds": round(elapsed, 3), **scores}
        )
        LOGGER.info("%s: F1 macro validación=%.4f", name, scores["f1_macro"])

    comparison.sort(key=lambda r: (r["f1_macro"], r["accuracy"]), reverse=True)
    best_name = str(comparison[0]["model"])
    best_pipeline = build_pipeline(best_name, seed=seed)

    cv_result: dict[str, Any] | None = None
    if cv_folds:
        cross_validation = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=seed
        )
        cv_scores = cross_val_score(
            best_pipeline, dev["text"], dev["label"],
            scoring="f1_macro", cv=cross_validation, n_jobs=1,
        )
        cv_result = {
            "folds": cv_folds,
            "f1_macro_mean": float(np.mean(cv_scores)),
            "f1_macro_std": float(np.std(cv_scores)),
        }

    # Reentrenar en train + validation tras finalizar la selección.
    best_pipeline.fit(dev["text"], dev["label"])
    started_test = time.perf_counter()
    prediction = best_pipeline.predict(test["text"])
    inference_seconds = time.perf_counter() - started_test
    test_scores = score_predictions(test["label"], prediction)
    matrix = confusion_matrix(test["label"], prediction, labels=[0, 1]).tolist()
    inference_benchmark = benchmark_inference(best_pipeline, test["text"], seed=seed)
    text_lengths = frame["text"].str.len()

    report: dict[str, Any] = {
        "version": "0.5.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_origin": ("DEMOSTRACIÓN SINTÉTICA" if demo else
                        "Sentiment140 Stanford — archivo oficial verificado"
                        if source_verification and source_verification.get("status") == "official_verified"
                        else "CSV local SIN verificación del origen"),
        "source_verification": {"status": "synthetic"} if demo else (source_verification or {"status": "unverified_local_csv"}),
        "demo": demo,
        "seed": seed,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.system(),
            "cpu_logical": os.cpu_count(),
            "library_sklearn": __import__("sklearn").__version__,
        },
        "audit": {
            **audit.as_dict(),
            "clean_negative_pct": round(100 * audit.clean_labels["negativo"] / audit.clean_rows, 2),
            "clean_positive_pct": round(100 * audit.clean_labels["positivo"] / audit.clean_rows, 2),
            "clean_text_length_mean": round(float(text_lengths.mean()), 2),
            "clean_text_length_median": round(float(text_lengths.median()), 2),
        },
        "splits": {"train": len(train), "validation": len(validation), "test": len(test)},
        "baseline_validation": baseline_metrics,
        "models_validation": comparison,
        "selected_model": best_name,
        "selection_metric": "f1_macro",
        "cross_validation_on_train_validation": cv_result,
        "test": {
            **test_scores,
            "confusion_matrix": matrix,
            "classification_report": classification_report(
                test["label"],
                prediction,
                labels=[0, 1],
                target_names=["negativo", "positivo"],
                output_dict=True,
                zero_division=0,
            ),
            "inference_seconds_batch": round(inference_seconds, 4),
            "inference_items": len(test),
            "batch_throughput_tweets_per_s": round(len(test) / max(inference_seconds, 1e-9), 2),
            "local_inference_benchmark": inference_benchmark,
        },
        "total_train_seconds": round(time.perf_counter() - start, 2),
    }
    with tempfile.TemporaryDirectory(prefix="nexus-", dir=out) as tmp_name:
        tmp = Path(tmp_name)
        joblib.dump(best_pipeline, tmp / MODEL_FILE)
        (tmp / REPORT_FILE).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _save_charts(tmp, audit.clean_labels, matrix)
        for name in files:
            shutil.move(str(tmp / name), str(out / name))

    LOGGER.info(
        "Modelo final: %s | F1 prueba=%.4f | artifacts=%s",
        best_name, test_scores["f1_macro"], out,
    )
    return report


def train_from_csv(
    data_path: Path, output_dir: Path, max_rows: int | None = 50_000,
    seed: int = 42, cv_folds: int = 3, force: bool = False, demo: bool = False,
) -> dict[str, Any]:
    """Audita, limpia, entrena y evalúa con una única llamada explícita."""
    frame, audit = load_sentiment140(data_path, max_rows=max_rows, seed=seed)
    provenance = {"status": "synthetic"} if demo else describe_provenance(data_path, audit.source_sha256)
    return fit_experiment(
        frame, audit, output_dir, seed=seed, cv_folds=cv_folds,
        force=force, demo=demo, source_verification=provenance,
    )


def load_artifacts(output_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Carga artefactos locales de confianza generados por esta aplicación.

    Advertencia: joblib no es seguro para archivos externos no confiables.
    """
    model_path = output_dir / MODEL_FILE
    report_path = output_dir / REPORT_FILE
    if not model_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Modelo y reporte no disponibles.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return joblib.load(model_path), report


def classify(model: Any, text: str) -> str:
    """Infiere sentimiento; no inventa probabilidades para SVM sin calibrar."""
    cleaned = normalize_tweet(text)
    if not cleaned:
        raise ValueError("El texto está vacío después de normalizarlo.")
    value = int(model.predict([cleaned])[0])
    if value not in (0, 1):
        raise ValueError("El modelo devolvió una etiqueta desconocida.")
    return "negativo" if value == 0 else "positivo"
