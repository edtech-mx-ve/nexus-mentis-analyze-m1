"""Inspección didáctica del TF-IDF REAL, sin reentrenar ni guardar tweets.

La traza usa el vectorizador ajustado del clasificador ACTIVO. Los pesos
TF-IDF no son atribuciones causales ni probabilidades de sentimiento.
"""
from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from nexus_mentis.data import normalize_tweet

MAX_TOKENS = 36
MAX_NGRAMS = 24
MAX_WEIGHTS = 14


def trace_active_tfidf(model: Any, text: str) -> dict[str, Any] | None:
    """Describe el preprocesado, analizador y vector exactos del modelo activo.

    None indica que el pipeline no tiene un TF-IDF de palabras identificado;
    nunca inventa una representación cuando hay un transformador distinto.
    Se presupone que el artefacto proviene de la instalación local confiable.
    """
    if not isinstance(text, str):
        raise TypeError("El texto debe ser una cadena.")
    if not 1 <= len(text) <= 5000:
        raise ValueError("El texto debe tener de 1 a 5000 caracteres.")
    if not isinstance(model, Pipeline):
        return None
    if not model.steps or model.steps[0][0] != "tfidf":
        return None
    vectorizer = model.steps[0][1]
    if not isinstance(vectorizer, TfidfVectorizer) or vectorizer.analyzer != "word":
        return None
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
        raise ValueError("Vectorizador TF-IDF sin ajustar.")

    normalized = normalize_tweet(text)
    if not normalized:
        raise ValueError("El texto quedo vacio despues de normalizar.")
    prepared = vectorizer.build_preprocessor()(normalized)
    # Usa el analizador real; refleja el token_pattern y los n-gramas configurados.
    analyzed = vectorizer.build_analyzer()(normalized)
    tokens = vectorizer.build_tokenizer()(prepared)
    matrix = vectorizer.transform([normalized]).tocsr()
    names = vectorizer.get_feature_names_out()
    if matrix.shape != (1, len(names)):
        raise ValueError("Dimension de TF-IDF inconsistente.")

    nonzero = sorted(
        ((int(idx), float(weight)) for idx, weight in zip(matrix.indices, matrix.data)),
        key=lambda item: (-item[1], str(names[item[0]])),
    )
    top_weights = [
        {"termino": str(names[idx]), "peso_tfidf": round(weight, 6)}
        for idx, weight in nonzero[:MAX_WEIGHTS]
    ]
    ngram_range = [int(v) for v in vectorizer.ngram_range]
    return {
        "disponible": True,
        "vectorizador": "TF-IDF ajustado del modelo activo",
        "modelo_vectorial": type(vectorizer).__name__,
        "texto_normalizado": normalized,
        "texto_preprocesado": prepared,
        "tokens": tokens[:MAX_TOKENS],
        "tokens_total": len(tokens),
        "ngrams": analyzed[:MAX_NGRAMS],
        "ngrams_total": len(analyzed),
        "rango_ngramas": ngram_range,
        "dimension_vector": int(matrix.shape[1]),
        "caracteristicas_no_cero": int(matrix.nnz),
        "pesos_principales": top_weights,
        "nota": (
            "Tokens y n-gramas del vectorizador real. TF-IDF mide peso "
            "de terminos, no probabilidad, causalidad ni contribucion firmada "
            "a la clase predicha. Los pesos nulos no se listan."
        ),
    }
