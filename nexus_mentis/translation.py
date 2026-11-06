"""Traducción local ES→EN para inferencia, aislada del modelo de sentimientos.

El traductor no descarga paquetes durante solicitudes HTTP. Un administrador
instala explícitamente el paquete con `setup-translation` antes de servir la app.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent.parent


class TranslationUnavailableError(RuntimeError):
    """No hay un modelo local español→inglés instalable o disponible."""


class TranslationFailureError(RuntimeError):
    """La traducción no pudo completarse de forma confiable."""


class Translator(Protocol):
    """Interfaz que permite pruebas aisladas sin paquetes grandes."""

    def is_available(self) -> bool:
        """Indica si existe una ruta de traducción lista para usar."""
        ...

    def translate(self, text: str) -> str:
        """Devuelve texto en inglés sin modificar el argumento."""
        ...


def translation_directory() -> Path:
    """Determina una ubicación local reproducible para el paquete lingüístico."""
    configured = os.environ.get("NEXUS_TRANSLATION_DIR")
    directory = Path(configured) if configured else PROJECT_DIR / "artifacts" / "translation_models"
    return directory.expanduser().resolve()


def _load_argos() -> tuple[Any, Any]:
    """Carga Argos sin exponer textos del usuario en bitácoras INFO."""
    # La librería registra entradas y traducciones completas a nivel INFO.
    # Se configuran sus loggers antes de importar y en cada solicitud.
    logging.getLogger("argostranslate").setLevel(logging.WARNING)
    logging.getLogger("argostranslate.utils").setLevel(logging.WARNING)
    os.environ.setdefault("ARGOS_PACKAGES_DIR", str(translation_directory()))
    try:
        from argostranslate import package, translate
    except (ImportError, OSError) as error:
        raise TranslationUnavailableError(
            "Falta el motor de traducción local. Ejecuta PREPARAR_TRADUCTOR_ES_EN.ps1."
        ) from error
    return package, translate


@dataclass(frozen=True)
class ArgosSpanishEnglish:
    """Servicio local de traducción únicamente desde español hacia inglés."""

    def is_available(self) -> bool:
        """Verifica que el paquete ES→EN esté instalado, sin consultar Internet."""
        try:
            _, argos_translate = _load_argos()
            langs = {lang.code: lang for lang in argos_translate.get_installed_languages()}
            if "es" not in langs or "en" not in langs:
                return False
            return langs["es"].get_translation(langs["en"]) is not None
        except (TranslationUnavailableError, RuntimeError, ValueError, OSError, KeyError):
            return False

    def translate(self, text: str) -> str:
        """Traduce una entrada no vacía; los fallos nunca se silencian."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("No hay texto español para traducir.")
        _, argos_translate = _load_argos()
        if not self.is_available():
            raise TranslationUnavailableError(
                "El paquete de idioma español→inglés no está instalado."
            )
        try:
            result = argos_translate.translate(text, "es", "en")
        except (RuntimeError, ValueError, OSError) as error:
            LOGGER.warning("Error del motor de traducción local: %s", type(error).__name__)
            raise TranslationFailureError("La traducción local falló.") from error
        if not isinstance(result, str) or not result.strip():
            raise TranslationFailureError("El traductor devolvió un texto vacío.")
        if len(result) > 5000:
            raise TranslationFailureError("La traducción excede el límite interno.")
        return result.strip()


def install_translation_model() -> dict[str, Any]:
    """Descarga e instala ES→EN desde el índice Argos, fuera de peticiones web.

    Requiere acceso HTTPS, dependencias instaladas y una decisión explícita
    del administrador. Registra procedencia, versión y SHA256 del paquete.
    """
    dest = translation_directory()
    dest.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ARGOS_PACKAGE_DIR", str(dest))
    # El equipo Windows puede confiar en certificados de empresa no incluidos
    # en certifi: truststore utiliza el almacén confiable del propio sistema.
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        LOGGER.warning("No está disponible truststore; se usará TLS del sistema Python.")

    package, _ = _load_argos()
    installed = [
        p for p in package.get_installed_packages()
        if p.from_code == "es" and p.to_code == "en" and p.type == "translate"
    ]
    if installed and ArgosSpanishEnglish().is_available():
        result = {
            "status": "already_installed",
            "engine": "Argos Translate",
            "pair": "es-en",
            "version": str(installed[0].package_version),
            "storage": str(dest),
        }
        LOGGER.info("El paquete español→inglés ya está disponible.")
        return result

    LOGGER.info("Consultando índice de modelos Argos para es→en.")
    package.update_package_index()
    candidates = [
        item for item in package.get_available_packages()
        if item.from_code == "es" and item.to_code == "en" and item.type == "translate"
    ]
    if not candidates:
        raise TranslationUnavailableError(
            "No se encontró modelo es→en en el índice Argos. Comprueba HTTPS o la conexión."
        )
    candidate = candidates[0]  # índice controlado por Argos; no usamos rutas del usuario
    LOGGER.info("Descargando paquete ES→EN (%s) por HTTPS.", candidate.package_version)
    downloaded = Path(candidate.download()).resolve()
    if downloaded.suffix != ".argosmodel" or not downloaded.is_file():
        raise TranslationFailureError("Paquete de traducción descargado inválido.")
    if downloaded.stat().st_size > 600 * 1024 * 1024:
        raise TranslationFailureError("El paquete de traducción supera el límite permitido.")
    hasher = hashlib.sha256()
    with downloaded.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    sha256 = hasher.hexdigest()
    package.install_from_path(downloaded)
    service = ArgosSpanishEnglish()
    if not service.is_available():
        raise TranslationFailureError("El paquete se instaló pero no puede traducir es→en.")
    # Comprobación funcional con ejemplo fijo, sin entradas de usuarios.
    sample = service.translate("El servicio fue excelente.")
    if not sample.strip():
        raise TranslationFailureError("El modelo de traducción no superó la comprobación local.")
    manifest = {
        "status": "installed",
        "engine": "Argos Translate",
        "pair": "es-en",
        "version": str(candidate.package_version),
        "package_sha256": sha256,
        "storage": str(dest),
        "note": "SHA-256 calculado localmente; no representa una firma del editor.",
    }
    # No registrar textos de usuarios ni credenciales.
    target = dest.parent / "translation_es_en_manifest.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    LOGGER.info("Paquete ES→EN instalado y comprobado; SHA-256=%s.", sha256)
    return manifest
