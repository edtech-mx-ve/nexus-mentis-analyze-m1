"""Entrada de nube: API intacta y readiness estricto del modelo y traductor."""
from __future__ import annotations
from fastapi import HTTPException
from nexus_mentis.api import app


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    if app.state.model is None or app.state.report is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    if not app.state.translator.is_available():
        raise HTTPException(status_code=503, detail="Traduccion ES-EN no disponible")
    return {"status": "ready", "module": "NEXUS ANALYZE"}
