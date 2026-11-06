"""Validacion de compilacion, inferencia y traduccion, sin dataset ni reentrenamiento."""
from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger("nexus.cloud.preflight")


def main() -> int:
    from nexus_mentis.api import app
    from nexus_mentis.model import classify
    from cloud_entry import healthz

    output = Path(os.environ["NEXUS_ARTIFACT_DIR"])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    if report.get("demo") is not False:
        raise ValueError("No se puede publicar un modelo DEMO")
    if report.get("source_verification", {}).get("status") != "official_verified":
        raise ValueError("No se verifico el origen de Sentiment140")
    if app.state.model is None or app.state.report is None:
        raise ValueError("No se cargo modelo persistido")
    if not app.state.translator.is_available():
        raise ValueError("Traductor local ES->EN no disponible")
    home = Path(__file__).parent / "nexus_mentis" / "web" / "modulo1.html"
    if not home.is_file():
        raise ValueError("Falta interfaz didactica")
    if classify(app.state.model, "This is a wonderful movie") not in ("positivo", "negativo"):
        raise ValueError("Prediccion invalida")
    translated = app.state.translator.translate("La pelicula fue excelente.")
    if not translated.strip():
        raise ValueError("Traduccion vacia")
    assert healthz()["status"] == "ready"
    LOGGER.info("Preflight OK: modelo real, ES->EN, interfaz y healthz")
    print("NEXUS CLOUD: preflight OK")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as exc:
        LOGGER.error("Preflight fallo: %s", type(exc).__name__)
        sys.exit(1)
