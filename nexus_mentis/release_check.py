"""Auditoría reproducible de preparación técnica NEXUS ANALYZE v0.9.

No certifica exactitud en español ni concede autorización de publicación.
Compara el reporte bilingüe con el modelo activo y el conjunto de diagnóstico.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus_mentis.bilingual_eval import DEFAULT_CASES, load_cases
from nexus_mentis.model import load_artifacts
from nexus_mentis.translation import ArgosSpanishEnglish

LOGGER = logging.getLogger(__name__)
VERSION = "0.9"
MAX_REPORT_BYTES = 4_000_000


def sha256_file(path: Path) -> str:
    """SHA256 incremental de un archivo local."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_report(path: Path) -> dict[str, Any]:
    """Lee JSON validado en tamaño y estructura, sin ejecutar código."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el reporte: {path.name}")
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("El reporte excede el límite permitido.")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("El reporte JSON no puede interpretarse.") from error
    if not isinstance(content, dict):
        raise ValueError("El reporte debe ser un objeto JSON.")
    return content


def evaluate_readiness(
    model_report: dict[str, Any],
    bilingual_report: dict[str, Any] | None,
    *,
    model_digest: str,
    cases_digest: str,
    translator_ready: bool,
) -> dict[str, Any]:
    """Valida evidencia, emite diagnóstico y bloquea aprobación pública automática.

    Reglas explícitas: una muestra sintética no prueba precisión poblacional.
    La preparación para *staging* es técnica, no un umbral de calidad del ML.
    """
    checks: dict[str, bool] = {}
    checks["real_model"] = model_report.get("demo") is False
    checks["source_verified"] = (
        isinstance(model_report.get("source_verification"), dict)
        and model_report["source_verification"].get("status") == "official_verified"
    )
    checks["translation_ready"] = bool(translator_ready)
    checks["bilingual_report_present"] = isinstance(bilingual_report, dict)
    bilingual = bilingual_report if isinstance(bilingual_report, dict) else {}
    checks["bilingual_schema_valid"] = bilingual.get("schema_version") == 1
    checks["report_matches_active_model"] = bilingual.get("model_report_sha256") == model_digest
    checks["cases_dataset_matches"] = bilingual.get("case_dataset_sha256") == cases_digest
    total = bilingual.get("total_cases")
    binary = bilingual.get("binary_cases")
    neutral = bilingual.get("neutral_cases")
    coverage = bilingual.get("coverage")
    checks["diagnostic_complete"] = (
        type(total) is int and total >= 40 and type(binary) is int and binary >= 30
        and type(neutral) is int and neutral >= 1 and binary + neutral == total
    )
    checks["coverage_at_least_90pct"] = (
        isinstance(coverage, (float, int)) and not isinstance(coverage, bool)
        and 0.90 <= coverage <= 1.0
    )
    score = bilingual.get("translated_binary_scores")
    checks["binary_metrics_present"] = (
        isinstance(score, dict)
        and isinstance(score.get("f1_macro"), (float, int))
        and 0 <= score["f1_macro"] <= 1
        and isinstance(score.get("confusion_matrix"), list)
    )
    paired = bilingual.get("english_reference_paired_binary_scores")
    checks["paired_comparison_present"] = (
        isinstance(paired, dict)
        and isinstance(score, dict)
        and paired.get("n") == score.get("n")
        and isinstance(paired.get("f1_macro"), (float, int))
        and isinstance(bilingual.get("paired_f1_delta_points_es_minus_en"), (float, int))
    )
    technical_ready = all(checks.values())
    issues = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "release_version": VERSION,
        "status": "staging_ready" if technical_ready else "blocked",
        "technical_staging_ready": technical_ready,
        "public_release_approved": False,
        "checks": checks,
        "blocking_issues": issues,
        "metrics": {
            "model_selected": model_report.get("selected_model"),
            "english_test_f1_macro": (
                model_report.get("test", {}).get("f1_macro")
                if isinstance(model_report.get("test"), dict) else None
            ),
            "diagnostic_es_f1_macro": (
                score.get("f1_macro") if isinstance(score, dict) else None
            ),
            "diagnostic_en_f1_macro": (
                paired.get("f1_macro") if isinstance(paired, dict) else None
            ),
            "coverage": coverage,
            "binary_coverage": bilingual.get("binary_coverage"),
            "paired_f1_delta_points_es_minus_en": bilingual.get(
                "paired_f1_delta_points_es_minus_en"
            ),
            "total_cases": total,
            "binary_cases": binary,
            "neutral_cases": neutral,
            "translation_vs_reference_disagreements": bilingual.get(
                "translation_vs_english_disagreements"
            ),
            "mean_es_pipeline_ms": (
                bilingual.get("timings", {}).get("spanish_pipeline", {}).get("mean_ms")
                if isinstance(bilingual.get("timings"), dict)
                and isinstance(bilingual["timings"].get("spanish_pipeline"), dict) else None
            ),
        },
        "review_required": [
            "Las etiquetas diagnósticas no fueron anotadas ciegamente por expertos independientes.",
            "Debe evaluarse un conjunto español representativo e independiente.",
            "Revisar manualmente traducciones, ironía, negaciones y casos bloqueados.",
            "Comprobar HTTPS, límites de solicitudes, privacidad y permisos en el host público.",
            "Aprobar condiciones de uso y licencias de datos y paquetes antes de publicar.",
        ],
        "scientific_scope": (
            "Los casos bilingües son sintéticos y curados; el F1 español es descriptivo, "
            "solo de casos binarios clasificados. No generalizable a la población."
        ),
    }


def prepare_release(
    model_dir: Path,
    bilingual_report_path: Path,
    cases_path: Path,
    *,
    translator_ready: bool,
) -> dict[str, Any]:
    """Relaciona evidencias locales con el modelo real cargable."""
    model_dir = model_dir.expanduser().resolve()
    model_report_path = model_dir / "report.json"
    model_report = read_report(model_report_path)
    # Verifica también que el artefacto de inferencia es cargable (solo fuentes confiables).
    loaded_model, _ = load_artifacts(model_dir)
    if loaded_model is None:
        raise ValueError("No se pudo cargar el modelo entrenado.")
    cases = load_cases(cases_path)
    bilingual = (
        read_report(bilingual_report_path) if bilingual_report_path.is_file() else None
    )
    result = evaluate_readiness(
        model_report, bilingual,
        model_digest=sha256_file(model_report_path),
        cases_digest=sha256_file(cases_path),
        translator_ready=translator_ready,
    )
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["model_report_sha256"] = sha256_file(model_report_path)
    result["case_dataset_sha256"] = sha256_file(cases_path)
    result["case_rows_verified"] = len(cases)
    return result


def render_markdown(report: dict[str, Any]) -> str:
    """Resumen técnico legible, sin tweets ni rutas personales."""
    metrics = report["metrics"]
    def fmt(value: Any, *, pct: bool = False) -> str:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return "No disponible"
        return f"{100 * value:.2f} %" if pct else str(value)

    checks = "\n".join(
        f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}"
        for name, passed in report["checks"].items()
    )
    pending = "\n".join(f"- {issue}" for issue in report["review_required"])
    return (
        "# NEXUS ANALYZE v0.9 — Auditoría de preparación\n\n"
        f"Estado técnico para staging: **{report['status']}**. "
        "**No autorizado para publicación pública automáticamente.**\n\n"
        "## Evidencia\n\n"
        f"- Modelo: {metrics['model_selected']}\n"
        f"- F1 macro test inglés Sentiment140: {fmt(metrics['english_test_f1_macro'], pct=True)}\n"
        f"- F1 macro diagnóstico ES→EN: {fmt(metrics['diagnostic_es_f1_macro'], pct=True)}\n"
        f"- F1 macro referencias inglesas: {fmt(metrics['diagnostic_en_f1_macro'], pct=True)}\n"
        f"- Cobertura de diagnóstico: {fmt(metrics['coverage'], pct=True)}\n"
        f"- Cobertura binaria: {fmt(metrics['binary_coverage'], pct=True)}\n"
        f"- Diferencia pareada F1 ES→EN menos EN (pp): "
        f"{fmt(metrics['paired_f1_delta_points_es_minus_en'])}\n"
        f"- Casos totales / binarios / neutrales: {metrics['total_cases']} / "
        f"{metrics['binary_cases']} / {metrics['neutral_cases']}\n"
        f"- Desacuerdos ES→EN frente a inglés de referencia: "
        f"{metrics['translation_vs_reference_disagreements']}\n"
        f"- Latencia media ES→EN+modelo sin red: "
        f"{fmt(metrics['mean_es_pipeline_ms'])} ms\n\n"
        "## Comprobaciones automáticas\n\n" + checks + "\n\n"
        "## Revisión imprescindible antes de publicar\n\n" + pending + "\n\n"
        "## Limitaciones\n\n" + report["scientific_scope"] + "\n"
    )


def persist_release(report: dict[str, Any], out_dir: Path, *, replace: bool = False) -> Path:
    """Guarda JSON y Markdown; no sobrescribe sin respaldo previo."""
    target = out_dir.expanduser().resolve()
    if target.name in {"data", "tests", ".venv"}:
        raise ValueError("No se permite usar una carpeta reservada como salida.")
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "readiness.json"
    md_path = target / "INFORME_VALIDACION_v09.md"
    if (json_path.exists() or md_path.exists()) and not replace:
        raise FileExistsError("El reporte existe. Usa --replace para crear respaldo.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for item in (json_path, md_path):
        if item.exists():
            item.replace(target / f"{item.stem}_backup_{timestamp}{item.suffix}")
    tmp_json = target / ".readiness.json.tmp"
    tmp_md = target / ".INFORME_VALIDACION_v09.md.tmp"
    try:
        with tmp_json.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        with tmp_md.open("x", encoding="utf-8") as handle:
            handle.write(render_markdown(report))
        os.replace(tmp_json, json_path)
        os.replace(tmp_md, md_path)
    finally:
        tmp_json.unlink(missing_ok=True)
        tmp_md.unlink(missing_ok=True)
    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Comprobación técnica de release NEXUS ANALYZE.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bilingual-report", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        readiness = prepare_release(
            args.model, args.bilingual_report, args.cases,
            translator_ready=ArgosSpanishEnglish().is_available(),
        )
        result = persist_release(readiness, args.out, replace=args.replace)
    except (FileNotFoundError, ValueError, OSError) as error:
        LOGGER.error("%s", error)
        return 2
    print(json.dumps({
        "status": readiness["status"],
        "report": str(result),
        "blocking_issues": readiness["blocking_issues"],
        "public_release_approved": False,
    }, ensure_ascii=False, indent=2))
    return 0 if readiness["technical_staging_ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
