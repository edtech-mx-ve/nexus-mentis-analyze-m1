"""Comandos locales para auditar, entrenar y probar NEXUS ANALYZE."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from nexus_mentis.data import load_sentiment140
from nexus_mentis.model import train_from_csv
from nexus_mentis.source import install_official_dataset, describe_provenance
from nexus_mentis.translation import (
    install_translation_model, TranslationUnavailableError, TranslationFailureError,
)

LOGGER = logging.getLogger(__name__)

NEGATIVE = [
    "My flight was cancelled without warning",
    "The airline lost my suitcase",
    "Waiting for hours with no explanation",
    "Terrible customer service at the gate",
    "I will never fly with this airline again",
    "The seats were dirty and uncomfortable",
    "Very rude staff today",
    "They charged unexpected fees",
    "Horrible experience with customer support",
    "The booking website did not work",
    "Delayed again and nobody helped",
    "I missed my connection because of delays",
    "No information about our departure",
    "Really disappointed in this flight",
    "The food was awful",
    "The plane was very late",
    "I am angry about the cancelled reservation",
    "What a frustrating airport experience",
    "Baggage service was painfully slow",
    "Unacceptable treatment of passengers",
]
POSITIVE = [
    "My flight arrived on time and was wonderful",
    "The cabin crew were helpful and friendly",
    "Excellent service at the airport",
    "I really enjoyed the comfortable seats",
    "What a smooth and pleasant trip",
    "Thank you for the amazing customer support",
    "The staff solved my problem quickly",
    "Very satisfied with this airline",
    "The flight attendants were fantastic",
    "Great experience from start to finish",
    "I love flying with this company",
    "Clean aircraft and delicious food",
    "The pilot made a perfect landing",
    "Such an easy and convenient booking process",
    "I was delighted by the attentive crew",
    "Wonderful trip with no delays",
    "My bag arrived safely and quickly",
    "Fast check in and excellent staff",
    "Best airline journey this year",
    "I am happy with the superb service",
]


def write_demo_csv(output: Path, force: bool = False) -> int:
    """Genera datos sintéticos solo para probar técnicamente el sistema."""
    destination = output.expanduser().resolve()
    if destination.suffix.lower() != ".csv":
        raise ValueError("El archivo de demostración debe terminar en .csv.")
    if destination.exists() and not force:
        raise FileExistsError("Ya existe el archivo de demostración. Usa --force.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    contexts = [
        "this morning at the airport",
        "on our recent journey",
        "during the weekend trip",
    ]
    rows: list[list[str | int]] = []
    for label, sentences in [(0, NEGATIVE), (4, POSITIVE)]:
        for sentence in sentences:
            for context in contexts:
                rows.append(
                    [label, len(rows) + 1, "2026-01-01", "NO_QUERY", "demo_user",
                     f"{sentence} {context}."]
                )
    with destination.open("w", encoding="latin-1", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return len(rows)


def create_parser() -> argparse.ArgumentParser:
    """Define una interfaz CLI documentada y reproducible."""
    parser = argparse.ArgumentParser(
        prog="nexus-mentis",
        description="NEXUS ANALYZE: auditoría, entrenamiento e inferencia local.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "fetch-sentiment140",
        help="Descarga el ZIP original desde Stanford, verifica SHA256 y extrae los CSV",
    )
    download.add_argument("--dest", type=Path, default=Path("data"))

    commands.add_parser(
        "setup-translation",
        help="Descarga una vez el modelo local Argos español→inglés (HTTPS)",
    )
    smoke = commands.add_parser(
        "smoke-bilingual",
        help="Comprueba traducción real y predicción con ejemplos fijos (no es evaluación científica)",
    )
    smoke.add_argument(
        "--artifacts", type=Path, default=Path("artifacts/sentiment140_real_200k")
    )

    demo = commands.add_parser("make-demo", help="Crea datos SINTÉTICOS para smoke tests")
    demo.add_argument("--output", type=Path, default=Path("data/demo_sentiment140.csv"))
    demo.add_argument("--force", action="store_true")

    audit = commands.add_parser("audit", help="Audita dataset real y muestra sin entrenar")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--max-rows", type=int, default=50_000)
    audit.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Entrena, compara y evalúa")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out", type=Path, default=Path("artifacts"))
    train.add_argument("--max-rows", type=int, default=50_000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--cv-folds", type=int, default=3)
    train.add_argument("--force", action="store_true")
    train.add_argument(
        "--demo", action="store_true",
        help="Etiqueta los resultados como DEMOSTRACIÓN (no académicos)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta subcomandos con errores controlados y logs de consola."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = create_parser().parse_args(argv)
    try:
        if args.command == "fetch-sentiment140":
            manifest = install_official_dataset(args.dest)
            print(
                f"ORIGEN OFICIAL VERIFICADO: {manifest['archive_sha256']}\\n"
                f"CSV de entrenamiento: {(args.dest / 'training.1600000.processed.noemoticon.csv').resolve()}\\n"
                "No se ha entrenado un modelo: ejecuta audit y train."
            )
        elif args.command == "setup-translation":
            result = install_translation_model()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("Traducción ES→EN lista. Reinicia Uvicorn si estaba ejecutándose.")
        elif args.command == "smoke-bilingual":
            from fastapi.testclient import TestClient
            from nexus_mentis.api import create_app
            with TestClient(create_app(args.artifacts)) as client:
                status = client.get("/api/health").json()
                if status["status"] != "ready":
                    raise ValueError("No hay modelo de sentimientos cargado.")
                if not status.get("translation", {}).get("es_to_en_ready"):
                    raise TranslationUnavailableError("El traductor local ES→EN no está listo.")
                examples = [
                    {"text": "Me encantó el servicio, fue excelente.", "language": "es"},
                    {"text": "El servicio fue terrible, no regresaré.", "language": "es"},
                    {"text": "The service was excellent.", "language": "en"},
                ]
                results: list[dict[str, object]] = []
                for payload in examples:
                    response = client.post("/api/predict", json=payload)
                    if response.status_code != 200:
                        raise ValueError(
                            f"Falló la inferencia de prueba: HTTP {response.status_code}"
                        )
                    answer = response.json()
                    if not answer.get("translated_text") or not answer.get("normalized_text"):
                        raise ValueError("Respuesta incompleta: falta traducción o normalización.")
                    results.append({
                        "language": answer["input_language"],
                        "translated_text": answer["translated_text"],
                        "sentiment": answer["sentiment"],
                        "translation_applied": answer["translation_applied"],
                        "total_ms": answer["total_ms"],
                    })
            print(json.dumps({
                "status": "smoke_passed",
                "warning": "Las tres frases son pruebas funcionales, NO miden exactitud en español.",
                "examples": results,
            }, indent=2, ensure_ascii=False))
        elif args.command == "make-demo":
            amount = write_demo_csv(args.output, force=args.force)
            print(f"Archivo DEMO (no real): {args.output} | {amount} tweets sintéticos")
        elif args.command == "audit":
            max_rows = None if args.max_rows == 0 else args.max_rows
            _, audit = load_sentiment140(args.data, max_rows, args.seed)
            payload = {
                "audit": audit.as_dict(),
                "source_verification": describe_provenance(args.data, audit.source_sha256),
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif args.command == "train":
            max_rows = None if args.max_rows == 0 else args.max_rows
            is_demo = args.demo or args.data.name.lower().startswith("demo_")
            report = train_from_csv(
                data_path=args.data,
                output_dir=args.out,
                max_rows=max_rows,
                seed=args.seed,
                cv_folds=args.cv_folds,
                force=args.force,
                demo=is_demo,
            )
            print(
                f"Modelo: {report['selected_model']} | "
                f"F1 macro prueba: {report['test']['f1_macro']:.4f} | "
                f"Accuracy prueba: {report['test']['accuracy']:.4f}"
            )
            if is_demo:
                print("ADVERTENCIA: métricas solo para validar el software; no son evidencia académica.")
            print(f"Resultados: {args.out.resolve() / 'report.json'}")
        return 0
    except (OSError, ValueError, TypeError, FileExistsError, TranslationUnavailableError, TranslationFailureError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
