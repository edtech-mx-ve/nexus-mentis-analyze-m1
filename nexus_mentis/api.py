"""API REST y servidor de interfaz local para NEXUS ANALYZE."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nexus_mentis.comparison_charts import CHART_NAMES, render_comparison_chart
from nexus_mentis.data import normalize_tweet
from nexus_mentis.linguistic_signals import inspect_signals
from nexus_mentis.translation_quality import guarded_translate, TranslationFidelityError
from nexus_mentis.insights import summarize_confusion, summarize_selection
from nexus_mentis.release_check import read_report, sha256_file
from nexus_mentis.bilingual_eval import DEFAULT_CASES
from nexus_mentis.model import classify, load_artifacts
from nexus_mentis.nlp_trace import trace_active_tfidf
from nexus_mentis.translation import (
    ArgosSpanishEnglish, Translator, TranslationUnavailableError,
    TranslationFailureError,
)

LOGGER = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parent / "web"
DEFAULT_ARTIFACTS = Path(os.getenv("NEXUS_ARTIFACT_DIR", "artifacts"))


class PredictionRequest(BaseModel):
    """Solicitud de inferencia con límites de longitud definidos."""

    text: str = Field(min_length=3, max_length=1000)
    language: Literal["es", "en"] = "en"  # compatibilidad con API v0.6


class PredictionResponse(BaseModel):
    """Predicción binaria sin probabilidad no calibrada."""

    sentiment: str
    model: str
    demo: bool
    inference_ms: float
    normalized_text: str
    original_text: str
    translated_text: str
    input_language: Literal["es", "en"]
    translation_applied: bool
    translation_engine: str | None
    translation_ms: float
    total_ms: float
    translation_quality_note: str | None = None
    interpretation_limited: bool = False
    linguistic_flags: list[str] = Field(default_factory=list)
    linguistic_notes: list[str] = Field(default_factory=list)
    source_emojis: list[str] = Field(default_factory=list)
    nlp_trace: dict[str, Any] | None = None
    nlp_trace_notice: str | None = None


def create_app(model_dir: Path | None = None, translator: Translator | None = None) -> FastAPI:
    """Construye una API con dependencias y artefactos configurables."""
    app = FastAPI(
        title="NEXUS MENTIS | NEXUS ANALYZE",
        version="0.9.5",
        description="API local de clasificación de sentimientos con auditoría trazable.",
    )
    directory = model_dir or DEFAULT_ARTIFACTS
    app.state.model = None
    app.state.translator = translator if translator is not None else ArgosSpanishEnglish()
    app.state.report = None
    try:
        app.state.model, app.state.report = load_artifacts(directory)
        LOGGER.info("Modelo local cargado (%s).", app.state.report["selected_model"])
    except (FileNotFoundError, ValueError, OSError) as error:
        LOGGER.warning("Aplicación sin modelo: %s", error)

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        """Limita carga de contenido, caching de textos y exposición del navegador."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'; "
            "form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        """Sirve la interfaz web accesible."""
        return FileResponse(WEB / "modulo1.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Comunica si la interfaz puede realizar inferencias."""
        report = app.state.report
        return {
            "status": "ready" if app.state.model is not None else "model_missing",
            "module": "NEXUS ANALYZE",
            "model": report["selected_model"] if report else None,
            "demo": bool(report["demo"]) if report else None,
            "source_verification": report.get("source_verification") if report else None,
            "translation": {
                "engine": "Argos Translate (local)",
                "es_to_en_ready": app.state.translator.is_available(),
                "downloads_during_requests": False,
            },
        }

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        """Expone estadísticas agregadas y métricas verificables del experimento."""
        report = app.state.report
        if report is None:
            raise HTTPException(status_code=503, detail="Modelo aún no entrenado.")
        test = report["test"]
        audit = report["audit"]
        return {
            "model": report["selected_model"],
            "demo": report["demo"],
            "data_origin": report["data_origin"],
            "source_verification": report.get("source_verification", {"status":"unverified_local_csv"}),
            "created_utc": report.get("created_utc"),
            "seed": report["seed"],
            "train_rows": report["splits"]["train"],
            "validation_rows": report["splits"]["validation"],
            "test_rows": report["splits"]["test"],
            "test_accuracy": test["accuracy"],
            "test_f1_macro": test["f1_macro"],
            "test_precision_macro": test.get("precision_macro"),
            "test_recall_macro": test.get("recall_macro"),
            "confusion_matrix": test["confusion_matrix"],
            "classification_report": test["classification_report"],
            "audited_source_rows": audit["source_rows"],
            "source_valid_rows": audit["source_valid_rows"],
            "clean_rows": audit["clean_rows"],
            "audit": audit,
            "validation_comparison": report["models_validation"],
            "baseline_validation": report["baseline_validation"],
            "cross_validation": report["cross_validation_on_train_validation"],
            "performance": test.get("local_inference_benchmark"),
            "batch_throughput_tweets_per_s": test.get("batch_throughput_tweets_per_s"),
            "environment": report.get("environment"),
            "selection_explanation": summarize_selection(
                report["models_validation"], report["baseline_validation"]
            ),
            "error_summary": summarize_confusion(test["confusion_matrix"]),
        }

    @app.get("/api/bilingual-evaluation")
    def bilingual_evaluation() -> dict[str, Any]:
        """Expone únicamente métricas realmente calculadas para el modelo activo."""
        output_dir = directory if directory.name == "artifacts" else directory.parent
        path = output_dir / "bilingual_eval_v08" / "report.json"
        if not path.is_file():
            return {
                "status": "not_run",
                "message": "La evaluación bilingüe aún no se ha ejecutado con el modelo real.",
            }
        if app.state.report is None:
            return {"status": "model_missing", "message": "No hay un modelo cargado."}
        source_report = directory / "report.json"
        if not source_report.is_file():
            return {"status": "model_missing", "message": "Falta el reporte del modelo."}
        try:
            report_digest = hashlib.sha256(source_report.read_bytes()).hexdigest()
            contents = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as error:
            LOGGER.warning("Reporte bilingüe inválido: %s", type(error).__name__)
            raise HTTPException(
                status_code=503, detail="No se puede leer la evaluación bilingüe."
            ) from error
        if contents.get("schema_version") != 1 or contents.get("model_report_sha256") != report_digest:
            return {"status": "stale", "message": "La evaluación pertenece a otro modelo o versión."}
        return {"status": "completed", "report": contents}

    @app.get("/api/tweet-regression")
    def tweet_regression() -> dict[str, Any]:
        """Muestra la regresión curada si corresponde al modelo y casos activos."""
        from nexus_mentis.tweet_regression import DEFAULT_CASES as TWEET_CASES

        if app.state.report is None:
            return {"status": "model_missing"}
        parent = directory if directory.name == "artifacts" else directory.parent
        report_path = parent / "tweet_regression_v091" / "report.json"
        model_report = directory / "report.json"
        if not report_path.is_file():
            return {"status": "not_run", "message": "La regresión de 10 tweets aún no se ha ejecutado."}
        if not model_report.is_file() or not TWEET_CASES.is_file():
            return {"status": "stale"}
        try:
            if report_path.stat().st_size > 150_000:
                return {"status": "invalid"}
            contents = json.loads(report_path.read_text(encoding="utf-8"))
            model_digest = hashlib.sha256(model_report.read_bytes()).hexdigest()
            case_digest = hashlib.sha256(TWEET_CASES.read_bytes()).hexdigest()
        except (OSError, ValueError, UnicodeError) as error:
            LOGGER.warning("Regresión de tweets ilegible: %s", type(error).__name__)
            return {"status": "invalid"}
        if not isinstance(contents, dict) or contents.get("schema_version") != 1:
            return {"status": "invalid"}
        if (contents.get("model_report_sha256") != model_digest
                or contents.get("cases_dataset_sha256") != case_digest):
            return {"status": "stale"}
        cases = contents.get("cases")
        if not isinstance(cases, list) or len(cases) != 10:
            return {"status": "invalid"}
        # Lista blanca: no emitir textos, traducciones, rutas locales ni campos inesperados.
        permitted = ("id", "group", "language", "expected", "predicted", "status", "flags", "total_ms", "translation_ms")
        safe_cases = [
            {name: record.get(name) for name in permitted}
            for record in cases if isinstance(record, dict)
        ]
        if len(safe_cases) != 10:
            return {"status": "invalid"}
        return {
            "status": "completed",
            "total_cases": contents.get("total_cases"),
            "eligible_binary_cases": contents.get("eligible_binary_cases"),
            "excluded_nonbinary_cases": contents.get("excluded_nonbinary_cases"),
            "blocked_cases": contents.get("blocked_cases"),
            "binary_scores_on_classified": contents.get("binary_scores_on_classified"),
            "timings": contents.get("timings"),
            "cases": safe_cases,
            "scope": "Casos curados para diagnóstico; no estiman la precisión general en español.",
        }

    @app.get("/api/controlled-study-v092")
    def controlled_study_v092() -> dict[str, Any]:
        """Comparación controlada con whitelist y vinculación al CSV activo."""
        from nexus_mentis.controlled_study import public_study_view

        if app.state.report is None:
            return {"status": "model_missing"}
        parent = directory if directory.name == "artifacts" else directory.parent
        target = parent / "controlled_study_v092" / "report.json"
        if not target.is_file():
            return {"status": "not_run"}
        try:
            if target.stat().st_size > 250_000:
                return {"status": "invalid"}
            document = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return {"status": "invalid"}
        return public_study_view(document, app.state.report)

    @app.get("/api/specialized-study-v093")
    def specialized_study_v093() -> dict[str, Any]:
        """Estudio exploratorio, vinculado al dataset activo y sin textos individuales."""
        from nexus_mentis.specialized_training import public_view

        if app.state.report is None:
            return {"status": "model_missing"}
        parent = directory if directory.name == "artifacts" else directory.parent
        target = parent / "specialized_study_v093" / "report.json"
        if not target.is_file():
            return {"status": "not_run"}
        try:
            if target.stat().st_size > 250_000:
                return {"status": "invalid"}
            document = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return {"status": "invalid"}
        return public_view(document, app.state.report)

    @app.get("/api/specialized-study-v094")
    def specialized_study_v094() -> dict[str, Any]:
        """Comparación exploratoria v0.9.4, sin textos ni cambios del modelo activo."""
        from nexus_mentis.specialized_training_v094 import public_view

        if app.state.report is None:
            return {"status": "model_missing"}
        parent = directory if directory.name == "artifacts" else directory.parent
        target = parent / "specialized_study_v094" / "report.json"
        if not target.is_file():
            return {"status": "not_run"}
        try:
            if target.stat().st_size > 250_000:
                return {"status": "invalid"}
            document = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return {"status": "invalid"}
        return public_view(document, app.state.report)

    @app.get("/api/release-readiness")
    def release_readiness() -> dict[str, Any]:
        """Estado técnico del release sin exponer rutas ni textos de prueba."""
        if app.state.report is None:
            return {"status": "model_missing"}
        model_path = directory / "report.json"
        parent = directory if directory.name == "artifacts" else directory.parent
        target = parent / "release_v09" / "readiness.json"
        if not target.is_file() or not model_path.is_file():
            return {"status": "not_run"}
        try:
            readiness = read_report(target)
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        except (ValueError, OSError, UnicodeError) as error:
            LOGGER.warning("Auditoría de publicación inválida: %s", type(error).__name__)
            return {"status": "invalid"}
        if readiness.get("schema_version") != 1 or readiness.get("model_report_sha256") != digest:
            return {"status": "stale"}
        if not DEFAULT_CASES.is_file() or readiness.get("case_dataset_sha256") != sha256_file(DEFAULT_CASES):
            return {"status": "stale"}
        checks = readiness.get("checks")
        if not isinstance(checks, dict) or not checks or not all(v is True for v in checks.values()):
            if readiness.get("status") == "staging_ready":
                return {"status": "invalid"}
        return {
            "status": readiness.get("status", "invalid"),
            "technical_staging_ready": readiness.get("technical_staging_ready") is True,
            "public_release_approved": False,
            "checks": readiness.get("checks", {}),
            "blocking_issues": readiness.get("blocking_issues", []),
            "scientific_scope": readiness.get("scientific_scope", ""),
            "metrics": readiness.get("metrics", {}),
        }

    @app.get("/api/charts/{chart_name}", include_in_schema=False)
    def chart(chart_name: str) -> Response:
        """Entrega gráficas históricas o comparativas basadas en métricas reales."""
        if chart_name in CHART_NAMES:
            if app.state.report is None:
                raise HTTPException(status_code=503, detail="Modelo aún no entrenado.")
            try:
                image = render_comparison_chart(app.state.report, chart_name)
            except ValueError as error:
                LOGGER.warning("Gráfico comparativo no disponible: %s", error)
                raise HTTPException(
                    status_code=422, detail="Reporte de validación inválido."
                ) from error
            return Response(
                content=image,
                media_type="image/png",
                headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            )

        allowed = {"class_distribution.png", "confusion_matrix.png"}
        if chart_name not in allowed:
            raise HTTPException(status_code=404, detail="Gráfico no encontrado.")
        path = directory / chart_name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Gráfico aún no disponible.")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        """Traduce ES→EN opcionalmente, luego clasifica texto inglés sin registrarlo."""
        report = app.state.report
        if app.state.model is None or report is None:
            raise HTTPException(status_code=503, detail="Modelo aún no entrenado.")

        original_text = payload.text.strip()
        if not original_text:
            raise HTTPException(status_code=422, detail="El texto no puede estar vacío.")
        translated_text = original_text
        translation_ms = 0.0
        translation_quality_note = None
        total_start = time.perf_counter_ns()
        if payload.language == "es":
            if not app.state.translator.is_available():
                raise HTTPException(
                    status_code=503,
                    detail="Traducción ES→EN no instalada. Ejecuta PREPARAR_TRADUCTOR_ES_EN.ps1.",
                )
            translate_start = time.perf_counter_ns()
            try:
                checked_translation = guarded_translate(original_text, app.state.translator)
                translated_text = checked_translation.text
                translation_quality_note = checked_translation.quality_note
            except TranslationFidelityError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            except TranslationUnavailableError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
            except TranslationFailureError as error:
                raise HTTPException(status_code=502, detail=str(error)) from error
            except (ValueError, TypeError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            translation_ms = (time.perf_counter_ns() - translate_start) / 1_000_000
            if not translated_text.strip():
                raise HTTPException(
                    status_code=502, detail="La traducción produjo texto vacío."
                )
            if len(translated_text) > 5000:
                raise HTTPException(status_code=422, detail="Traducción excesivamente larga.")

        assessment = inspect_signals(original_text, translated_text, payload.language)
        translated_text = assessment.translated_text

        try:
            infer_start = time.perf_counter_ns()
            sentiment = classify(app.state.model, translated_text)
            inference_ms = (time.perf_counter_ns() - infer_start) / 1_000_000
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        # Visualizacion opcional: nunca bloquea la clasificacion ya obtenida.
        nlp_trace = None
        trace_notice = None
        try:
            nlp_trace = trace_active_tfidf(app.state.model, translated_text)
            if nlp_trace is None:
                trace_notice = (
                    "El modelo activo no expone un TF-IDF de palabras compatible; "
                    "no se muestra una representacion inventada."
                )
        except (TypeError, ValueError, AttributeError, RuntimeError) as error:
            LOGGER.warning("Traza TF-IDF no disponible (%s).", type(error).__name__)
            trace_notice = "No fue posible inspeccionar el vectorizador del modelo activo."

        return PredictionResponse(
            sentiment=sentiment,
            model=report["selected_model"],
            demo=bool(report["demo"]),
            inference_ms=round(inference_ms, 3),
            normalized_text=normalize_tweet(translated_text),
            original_text=original_text,
            translated_text=translated_text,
            input_language=payload.language,
            translation_applied=payload.language == "es",
            translation_engine="Argos Translate (local)" if payload.language == "es" else None,
            translation_ms=round(translation_ms, 3),
            total_ms=round((time.perf_counter_ns() - total_start) / 1_000_000, 3),
            translation_quality_note=translation_quality_note,
            interpretation_limited=assessment.interpretation_limited,
            linguistic_flags=list(assessment.flags),
            linguistic_notes=list(assessment.notes),
            source_emojis=list(assessment.source_emojis),
            nlp_trace=nlp_trace,
            nlp_trace_notice=trace_notice,
        )

    return app


app = create_app()
