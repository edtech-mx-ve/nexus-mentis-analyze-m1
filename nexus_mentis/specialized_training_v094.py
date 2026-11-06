"""NEXUS ANALYZE v0.9.4: estudio exploratorio con lectura de cláusulas.

No utiliza tweets diagnósticos para entrenamiento ni para selección. El modelo
activo sigue intacto y la prueba compartida ya fue consultada en estudios
anteriores; sus resultados son evidencia exploratoria, no prueba nueva ciega.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from nexus_mentis.controlled_study import (
    NAMES, _comparison_test, _digest_texts, make_model, split_fixed, validate_frame,
)
from nexus_mentis.data import load_sentiment140, normalize_tweet
from nexus_mentis.model import score_predictions
from nexus_mentis.source import describe_provenance
from nexus_mentis.specialized_training import (
    DEFAULT_CASES as OLD_CASES, KNOWN_CASES, LABELS,
    load_curated_cases, verify_separation,
)

LOGGER = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent
NEW_CASES = PROJECT / "evaluation" / "specialized_cases_v094.json"
REFERENCE = "Referencia palabras+caracteres"
OLD = "Especializado v0.9.3"
NEW = "Candidato v0.9.4: cláusula final"
NAMES_V094 = (REFERENCE, OLD, NEW)
# Umbrales fijados antes de ejecutar el estudio. No consultar el test para elegir.
MAX_VALIDATION_LOSS = 0.002  # máximo 0,2 pp F1
MIN_TARGET_GAIN = 0.04      # mínimo 4 pp F1 en validación sintética nueva
OLD_SYNTH_WEIGHT = 40.0
NEW_SYNTH_WEIGHT = 35.0

_CONTRAST = re.compile(r"\b(?:but|yet|however|nevertheless|still)\b", flags=re.I)
_INITIAL = re.compile(r"^\s*(?:although|though|while)\b[^,;]{3,160}[,;]\s*", flags=re.I)


def extract_final_clause(texts: Any) -> list[str]:
    """Extrae cláusula final tras contraste, sin inventar ni alterar polaridad."""
    if isinstance(texts, (str, bytes)):
        raise TypeError("Se espera una colección de textos.")
    result: list[str] = []
    for raw in texts:
        if not isinstance(raw, str):
            raise TypeError("Todos los textos deben ser cadenas.")
        text = raw.strip()
        matches = list(_CONTRAST.finditer(text))
        if matches:
            clause = text[matches[-1].end():].strip(" ,;:-")
        elif _INITIAL.match(text):
            clause = _INITIAL.sub("", text, count=1).strip()
        else:
            clause = ""
        result.append(clause if len(clause) >= 3 else "")
    return result


def make_clause_aware_model(seed: int = 42) -> Pipeline:
    """TF-IDF palabras/caracteres y TF-IDF independiente para la cláusula final."""
    baseline = make_model(NAMES[1], seed)
    ordinary_features = baseline.named_steps["tfidf"]
    final_clause_features = Pipeline([
        ("extract", FunctionTransformer(extract_final_clause, validate=False)),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=12_000, min_df=2,
            sublinear_tf=True, strip_accents="unicode",
        )),
    ])
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("baseline", ordinary_features),
            ("final_clause", final_clause_features),
        ])),
        ("classifier", LogisticRegression(
            C=1.0, max_iter=500, random_state=seed
        )),
    ])


def combine_curated(old: dict[str, pd.DataFrame],
                    new: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Rechaza textos idénticos o diagnósticos copiados entre colecciones."""
    old_texts = set(pd.concat([f["text"] for f in old.values()]))
    new_texts = set(pd.concat([f["text"] for f in new.values()]))
    if old_texts & new_texts:
        raise ValueError("Los conjuntos v0.9.3 y v0.9.4 no son disjuntos.")
    return {
        split: pd.concat([old[split], new[split]], ignore_index=True)
        for split in ("train", "validation", "test")
    }


def fit_weighted(
    base: pd.DataFrame, old_train: pd.DataFrame,
    new_train: pd.DataFrame, *, seed: int, clause_aware: bool,
) -> Pipeline:
    """Ajusta sólo con train (+val general tras seleccionar); pesos predefinidos."""
    for frame in (base, old_train, new_train):
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError("Faltan columnas text y label.")
    inputs = pd.concat([
        base[["text", "label"]],
        old_train[["text", "label"]],
        new_train[["text", "label"]],
    ], ignore_index=True)
    weights = np.concatenate((
        np.ones(len(base), dtype=np.float64),
        np.full(len(old_train), OLD_SYNTH_WEIGHT),
        np.full(len(new_train), NEW_SYNTH_WEIGHT),
    ))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Pesos inválidos.")
    model = make_clause_aware_model(seed) if clause_aware else make_model(NAMES[1], seed)
    model.fit(inputs["text"], inputs["label"], classifier__sample_weight=weights)
    return model


def predict_metrics(model: Pipeline, frame: pd.DataFrame) -> dict[str, float]:
    return score_predictions(frame["label"], np.asarray(model.predict(frame["text"]), dtype=int))


def curated_group_scores(model: Pipeline, curated: pd.DataFrame) -> dict[str, Any]:
    predicted = np.asarray(model.predict(curated["text"]), dtype=int)
    expected = curated["label"].to_numpy(dtype=int)
    groups: dict[str, Any] = {}
    for group, subset in curated.groupby("group", sort=True):
        idx = curated.index.get_indexer(subset.index)
        groups[group] = {"n": len(idx), "correct": int((predicted[idx] == expected[idx]).sum())}
    return {"n": len(curated), "f1_macro": score_predictions(
        curated["label"], predicted)["f1_macro"], "by_group": groups}


def choose_on_validation(
    main: dict[str, dict[str, float]],
    targeted: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Nunca consulta prueba común ni tweets históricos."""
    first = main[REFERENCE]["f1_macro"]
    base_groups = targeted[REFERENCE]["by_group"]
    candidates: list[str] = []
    for name in (OLD, NEW):
        if main[name]["f1_macro"] < first - MAX_VALIDATION_LOSS:
            continue
        if targeted[name]["f1_macro"] - targeted[REFERENCE]["f1_macro"] < MIN_TARGET_GAIN:
            continue
        groups = targeted[name]["by_group"]
        if any(groups[g]["correct"] < base_groups[g]["correct"]
               for g in ("sarcasm_negative", "contrast_positive")):
            continue
        candidates.append(name)
    if not candidates:
        return REFERENCE, "Ningún candidato cumplió calidad general, mejora focal y conservación de grupos."
    winner = max(candidates, key=lambda n: (
        main[n]["f1_macro"],
        targeted[n]["f1_macro"],
    ))
    return winner, "Selección sólo con validación: pérdida F1 ≤0,2 pp, mejora focal ≥4 pp y sin regresión en grupos críticos."


def diagnose_known(models: dict[str, Pipeline], translator: Any | None) -> dict[str, Any]:
    """Mismos dos textos EN para los 3 modelos; traducción ES se realiza una vez."""
    if translator is None or not translator.is_available():
        return {"status": "not_run", "reason": "Traductor local ES→EN no disponible"}
    from nexus_mentis.tweet_regression import load_tweet_cases
    from nexus_mentis.translation_quality import guarded_translate, TranslationFidelityError
    from nexus_mentis.linguistic_signals import inspect_signals
    from nexus_mentis.translation import TranslationFailureError, TranslationUnavailableError
    cases = load_tweet_cases(KNOWN_CASES)
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case["id"] not in ("tweet04", "tweet05"):
            continue
        try:
            translated = guarded_translate(case["text"], translator)
            inspected = inspect_signals(case["text"], translated.text, "es")
            normalized = normalize_tweet(inspected.translated_text)
        except (TranslationFidelityError, TranslationFailureError, TranslationUnavailableError):
            rows.append({"id": case["id"], "status": "blocked"})
            continue
        predictions = {
            name: LABELS[int(model.predict([normalized])[0])]
            for name, model in models.items()
        }
        rows.append({
            "id": case["id"], "expected": case["expected"],
            "predictions": predictions,
            "correct": {name: pred == case["expected"] for name, pred in predictions.items()},
            "status": "classified",
        })
    return {
        "status": "completed" if len(rows) == 2 else "incomplete",
        "cases": rows,
        "note": "Casos revisados históricamente; exclusivamente diagnóstico post hoc, NO selección.",
    }


def run_study(frame: pd.DataFrame, old: dict[str, pd.DataFrame],
              new: dict[str, pd.DataFrame], *, seed: int = 42,
              bootstrap_repeats: int = 200, translator: Any | None = None
              ) -> tuple[dict[str, Any], Pipeline]:
    """Mismo split para tres modelos; la selección termina antes de mirar test."""
    validate_frame(frame)
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Semilla inválida.")
    if not 100 <= bootstrap_repeats <= 2000:
        raise ValueError("Bootstrap fuera de rango.")
    train, val, test = split_fixed(frame, seed)
    verify_separation((train, val, test), old)
    verify_separation((train, val, test), new)
    combined = combine_curated(old, new)
    timing: dict[str, float] = {}
    def train_models(base: pd.DataFrame, phase: str) -> dict[str, Pipeline]:
        models: dict[str, Pipeline] = {}
        for name in NAMES_V094:
            started = time.perf_counter()
            if name == REFERENCE:
                model = make_model(NAMES[1], seed)
                model.fit(base["text"], base["label"])
            elif name == OLD:
                model = fit_weighted(
                    base, old["train"], new["train"].iloc[:0], seed=seed, clause_aware=False
                )
            else:
                model = fit_weighted(
                    base, old["train"], new["train"], seed=seed, clause_aware=True
                )
            timing[f"{phase}_{name}"] = round(time.perf_counter() - started, 3)
            models[name] = model
        return models

    before = train_models(train, "train")
    main_val = {name: predict_metrics(model, val) for name, model in before.items()}
    targeted_val = {name: curated_group_scores(model, combined["validation"])
                    for name, model in before.items()}
    selected, reason = choose_on_validation(main_val, targeted_val)

    dev = pd.concat([train, val], ignore_index=True)
    final = train_models(dev, "refit")
    predictions = {
        name: np.asarray(model.predict(test["text"]), dtype=int)
        for name, model in final.items()
    }
    primary = _comparison_test(
        test["label"],
        {NAMES[0]: predictions[REFERENCE], NAMES[1]: predictions[NEW]},
        seed, bootstrap_repeats
    )
    main_test = {
        "scores": {name: score_predictions(test["label"], pred)
                   for name, pred in predictions.items()},
        "f1_delta_new_minus_reference_pp": primary["f1_delta_candidate_minus_baseline_pp"],
        "ci95_delta_new_minus_reference_pp": primary["f1_delta_bootstrap_ci95"],
        "shared_test_rows": len(test),
        "test_sha256": _digest_texts(test),
    }
    targeted_test = {
        name: curated_group_scores(model, combined["test"])
        for name, model in final.items()
    }
    known = diagnose_known(final, translator)
    result = {
        "schema_version": 1, "version": "v0.9.4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "rows_clean": len(frame),
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "curated": {
            "source": "150 textos sintéticos originales (100 v0.9.3 + 50 v0.9.4), NO tweets reales",
            "counts": {"train": len(combined["train"]),
                       "validation": len(combined["validation"]),
                       "test": len(combined["test"])},
            "old_weight": OLD_SYNTH_WEIGHT,
            "new_weight": NEW_SYNTH_WEIGHT,
            "validation": targeted_val,
            "test_details": targeted_test,
        },
        "main_validation": main_val,
        "selection": {
            "selected": selected, "reason": reason,
            "main_validation_max_loss_pp": MAX_VALIDATION_LOSS * 100,
            "min_curated_validation_gain_pp": MIN_TARGET_GAIN * 100,
            "decided_before_test": True,
        },
        "main_test": main_test,
        "known_failures": known,
        "timings_seconds": timing,
        "environment": {
            "python": sys.version.split()[0],
            "scikit_learn": __import__("sklearn").__version__,
            "platform": platform.system(),
        },
        "limitations": (
            "EXPLORATORIO. La prueba Sentiment140 y los 2 tweets históricos ya fueron "
            "observados en versiones previas: no constituyen una nueva evaluación ciega. "
            "Los ejemplos sintéticos no demuestran generalización a español, sarcasmo "
            "en producción ni Twitter/X actual. No hay sustitución automática."
        ),
    }
    return result, final[NEW]


def persist(report: dict[str, Any], model: Pipeline, directory: Path) -> Path:
    """Crea nuevo directorio completo; nunca sobrescribe resultados existentes."""
    target = directory.expanduser().resolve()
    if target.exists():
        raise FileExistsError("Ya existe la carpeta del experimento; se conserva.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nexus_v094_", dir=target.parent) as tmp:
        folder = Path(tmp)
        (folder / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        joblib.dump(model, folder / "candidate_v094.joblib")
        folder.rename(target)
    return target / "report.json"


def public_view(report: Any, active_report: dict[str, Any] | None) -> dict[str, Any]:
    """Sólo métricas agregadas; rechaza reportes incorrectos o de otros datasets."""
    if active_report is None:
        return {"status": "model_missing"}
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        return {"status": "invalid"}
    try:
        provenance = report["data"]["provenance"]
        if provenance["status"] != "official_verified":
            return {"status": "unverified"}
        if provenance["csv_sha256"] != active_report["source_verification"]["csv_sha256"]:
            return {"status": "stale"}
        names = list(NAMES_V094)
        scores = report["main_test"]["scores"]
        vals = report["main_validation"]
        for name in names:
            if any(not math.isfinite(float(k)) or not 0 <= float(k) <= 1
                   for k in (scores[name]["f1_macro"], vals[name]["f1_macro"])):
                return {"status": "invalid"}
        safe_cases = [
            {"id": row["id"], "expected": row.get("expected"),
             "predictions": {n: row.get("predictions", {}).get(n) for n in names}}
            for row in report.get("known_failures", {}).get("cases", [])
            if row.get("id") in ("tweet04", "tweet05") and row.get("status") == "classified"
        ]
        return {
            "status": "completed",
            "names": names,
            "selected": report["selection"]["selected"],
            "reason": report["selection"]["reason"],
            "validation_f1": {n: vals[n]["f1_macro"] for n in names},
            "test_f1": {n: scores[n]["f1_macro"] for n in names},
            "delta_new_vs_ref_pp": report["main_test"]["f1_delta_new_minus_reference_pp"],
            "ci95": report["main_test"]["ci95_delta_new_minus_reference_pp"],
            "known_cases": safe_cases,
            "note": "Exploratorio; no cambia el clasificador activo ni valida español.",
        }
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return {"status": "invalid"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEXUS ANALYZE v0.9.4 estudio exploratorio.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/specialized_study_v094"))
    parser.add_argument("--cases-old", type=Path, default=OLD_CASES)
    parser.add_argument("--cases-new", type=Path, default=NEW_CASES)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--allow-unverified", action="store_true",
                        help="Sólo para pruebas técnicas con datos sintéticos")
    parser.add_argument("--no-translate", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        if args.out.expanduser().resolve().exists():
            raise FileExistsError("Ya existe el experimento. No se sobrescribe.")
        if not 500 <= args.max_rows <= 1_600_000:
            raise ValueError("max-rows fuera de rango [500, 1600000].")
        frame, audit = load_sentiment140(args.data, max_rows=args.max_rows, seed=args.seed)
        provenance = describe_provenance(args.data, audit.source_sha256)
        if provenance.get("status") != "official_verified" and not args.allow_unverified:
            raise ValueError("Fuente no verificada.")
        old = load_curated_cases(args.cases_old)
        new = load_curated_cases(args.cases_new)
        translator = None
        if not args.no_translate:
            from nexus_mentis.translation import ArgosSpanishEnglish
            translator = ArgosSpanishEnglish()
        started = time.perf_counter()
        report, candidate = run_study(
            frame, old, new, seed=args.seed,
            bootstrap_repeats=args.bootstrap_repeats,
            translator=translator,
        )
        report["data"] = {
            "audit": audit.as_dict(),
            "provenance": provenance,
            "curated_old_sha256": hashlib.sha256(args.cases_old.read_bytes()).hexdigest(),
            "curated_new_sha256": hashlib.sha256(args.cases_new.read_bytes()).hexdigest(),
        }
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        path = persist(report, candidate, args.out)
        print(json.dumps({
            "status": "completed", "report": str(path),
            "selected_by_validation": report["selection"]["selected"],
            "validation_f1": {n: x["f1_macro"] for n, x in report["main_validation"].items()},
            "test_f1": {n: x["f1_macro"] for n, x in report["main_test"]["scores"].items()},
            "known_failures": report["known_failures"],
            "note": "Sin promover modelos; estudio exploratorio con test históricamente observado.",
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        LOGGER.error("Fallo en v0.9.4: %s: %s", type(exc).__name__, str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
