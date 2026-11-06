"""Descarga verificable y extracción segura del corpus oficial Sentiment140."""
from __future__ import annotations

import hashlib
import json
import logging
import ssl
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
OFFICIAL_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
# SHA-256 esperado registrado por NEXUS tras la verificación de la descarga; no se atribuye a un hash publicado por Stanford.
OFFICIAL_ARCHIVE_SHA256 = "004a3772c8a7ff9bbfeb875880f47f0679d93fc63e5cf9cff72d54a8a6162e57"
OFFICIAL_ARCHIVE_BYTES = 81_363_704
TRAIN_NAME = "training.1600000.processed.noemoticon.csv"
TEST_NAME = "testdata.manual.2009.06.14.csv"
MANIFEST_NAME = "sentiment140_source.json"
MAX_DOWNLOAD_BYTES = 150 * 1024 * 1024
MAX_EXTRACT_BYTES = 400 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Calcula SHA256 de un archivo mediante lectura incremental."""
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _download_verified(destination: Path) -> None:
    """Descarga el ZIP por HTTPS y verifica tamaño y SHA256 antes de conservarlo."""
    request = urllib.request.Request(
        OFFICIAL_URL,
        headers={"User-Agent": "NEXUS-MENTIS-ANALYZE/0.5 (academic research)"},
    )
    written = 0
    next_log_bytes = 10 * 1024 * 1024
    digest = hashlib.sha256()
    # x+b evita sobreescrituras accidentales; el temporal pertenece a esta ejecución.
    # truststore consulta el almacén de CA del sistema (Windows CryptoAPI).
    # No se desactiva la validación de certificados ni de nombres de host.
    try:
        import truststore
    except ImportError as exc:
        raise ValueError(
            "Falta la dependencia truststore. Ejecuta: python -m pip install truststore"
        ) from exc
    tls_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with urllib.request.urlopen(request, timeout=90, context=tls_context) as response:
        if not response.geturl().lower().startswith("https://"):
            raise ValueError("La fuente redirigió fuera de HTTPS; descarga cancelada.")
        with destination.open("xb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                if written > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Descarga excede el tamaño máximo autorizado.")
                digest.update(block)
                handle.write(block)
                if written >= next_log_bytes:
                    LOGGER.info("Descargados %.1f MB de aproximadamente 81.4 MB.", written / (1024 * 1024))
                    next_log_bytes += 10 * 1024 * 1024
    if written != OFFICIAL_ARCHIVE_BYTES or digest.hexdigest() != OFFICIAL_ARCHIVE_SHA256:
        raise ValueError(
            "La descarga no coincide con el archivo oficial: "
            f"tamaño={written}, SHA256={digest.hexdigest()}."
        )


def _extract_csv(archive: Path, output: Path, basename: str) -> tuple[str, int]:
    """Extrae un CSV conocido sin recorrer rutas del ZIP y con límite de tamaño."""
    if basename not in {TRAIN_NAME, TEST_NAME}:
        raise ValueError("Nombre de CSV no permitido.")
    hasher = hashlib.sha256()
    with zipfile.ZipFile(archive) as container:
        matches = [item for item in container.infolist() if Path(item.filename).name == basename]
        if len(matches) != 1:
            raise ValueError(f"No se encontró un único archivo {basename} en el ZIP.")
        item = matches[0]
        if item.is_dir() or item.file_size > MAX_EXTRACT_BYTES:
            raise ValueError("Entrada ZIP inválida o demasiado grande.")
        total = 0
        with container.open(item, "r") as stream, output.open("xb") as target:
            while block := stream.read(1024 * 1024):
                total += len(block)
                if total > MAX_EXTRACT_BYTES:
                    raise ValueError("El contenido extraído excede el límite.")
                hasher.update(block)
                target.write(block)
    return hasher.hexdigest(), total


def install_official_dataset(directory: Path) -> dict[str, Any]:
    """Instala exclusivamente los dos CSV originales sin sustituir archivos previos.

    Puede reutilizar un ZIP local solo si coincide exactamente con el SHA256 oficial.
    No crea un manifiesto verificado cuando la descarga/extracción falla.
    """
    folder = directory.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    destination_train = folder / TRAIN_NAME
    destination_test = folder / TEST_NAME
    manifest = folder / MANIFEST_NAME
    existing = [path for path in (destination_train, destination_test, manifest) if path.exists()]
    if existing:
        raise FileExistsError(
            "Existen archivos de Sentiment140: "
            + ", ".join(path.name for path in existing)
            + ". Se conservan; utiliza otra carpeta si necesitas reinstalar."
        )
    archive = folder / "trainingandtestdata.zip"
    keep_zip = archive.is_file()
    if keep_zip:
        if archive.stat().st_size != OFFICIAL_ARCHIVE_BYTES or sha256_file(archive) != OFFICIAL_ARCHIVE_SHA256:
            raise ValueError("El ZIP encontrado no coincide con el original verificado.")
        LOGGER.info("Se reutilizará el ZIP oficial existente.")
    else:
        # mismo directorio para renombrado atómico
        with tempfile.TemporaryDirectory(prefix="nexus-download-", dir=folder) as temp:
            staged = Path(temp) / "download.zip"
            LOGGER.info("Descargando Sentiment140 desde Stanford mediante HTTPS.")
            _download_verified(staged)
            staged.replace(archive)
    with tempfile.TemporaryDirectory(prefix="nexus-extract-", dir=folder) as temp:
        root = Path(temp)
        digest_train, size_train = _extract_csv(archive, root / TRAIN_NAME, TRAIN_NAME)
        digest_test, size_test = _extract_csv(archive, root / TEST_NAME, TEST_NAME)
        provenance: dict[str, Any] = {
            "dataset": "Sentiment140",
            "status": "official_verified",
            "source": OFFICIAL_URL,
            "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "archive_size_bytes": OFFICIAL_ARCHIVE_BYTES,
            "download_or_reuse_utc": datetime.now(timezone.utc).isoformat(),
            "files": {
                TRAIN_NAME: {"sha256": digest_train, "bytes": size_train},
                TEST_NAME: {"sha256": digest_test, "bytes": size_test},
            },
            "note": "La procedencia e integridad del ZIP se verificaron; las etiquetas siguen siendo debiles.",
        }
        # CSV y manifiesto solo se publican una vez que ambos CSV se extrajeron.
        (root / MANIFEST_NAME).write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Los destinos son exclusivos; nunca se reemplaza un dataset existente.
        for name in (TRAIN_NAME, TEST_NAME, MANIFEST_NAME):
            (root / name).rename(folder / name)
    LOGGER.info("Sentiment140 verificado e instalado: %s.", destination_train)
    return provenance


def describe_provenance(csv_path: Path, actual_sha256: str) -> dict[str, Any]:
    """Valida que el CSV utilizado sea el extraído de un ZIP de Stanford verificado."""
    path = csv_path.expanduser().resolve()
    manifest_path = path.parent / MANIFEST_NAME
    unverified: dict[str, Any] = {
        "status": "unverified_local_csv",
        "source": None,
        "csv_sha256": actual_sha256,
    }
    if not manifest_path.is_file():
        return unverified
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        file_info = doc["files"][path.name]
        valid = (doc.get("status") == "official_verified"
                 and doc.get("archive_sha256") == OFFICIAL_ARCHIVE_SHA256
                 and doc.get("source") == OFFICIAL_URL
                 and file_info.get("sha256") == actual_sha256
                 and file_info.get("bytes") == path.stat().st_size)
        if valid:
            return {
                "status": "official_verified",
                "source": OFFICIAL_URL,
                "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
                "csv_sha256": actual_sha256,
            }
    except (OSError, ValueError, KeyError, TypeError):
        LOGGER.warning("Manifiesto de origen inválido para %s.", path.name)
    return unverified
