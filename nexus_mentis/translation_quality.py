"""Reglas conservadoras de control de fidelidad para traducción ES→EN.

No constituyen validación semántica universal; solo verifican indicadores
lingüísticos de alto riesgo identificados en las pruebas del laboratorio.
Cuando no se puede preservar una expresión crítica, se rechaza la inferencia.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class LocalTranslator(Protocol):
    """Interfaz mínima para aplicar traducción sin efectos secundarios."""
    def translate(self, text: str) -> str:
        """Traduce texto español a inglés."""


class TranslationFidelityError(RuntimeError):
    """La traducción presenta una inversión potencial del significado."""


@dataclass(frozen=True)
class CheckedTranslation:
    """Traducción inspeccionada y su alcance de revisión."""

    text: str
    quality_note: str | None = None
    paraphrase_used: bool = False


_NEGATIVE_EN = re.compile(
    r"\b(bad|terrible|awful|poor|horrible|dreadful|lousy|worst|"
    r"disappointing|negative|atrocious|appalling|unpleasant)\b|"
    r"\bnot\s+(?:a\s+)?(?:very\s+)?(?:good|great|excellent)\b",
    re.IGNORECASE,
)
_SEVERE_ES = re.compile(
    r"\b(?:p[eé]sima\s+experiencia|p[eé]simo\s+servicio)\b",
    re.IGNORECASE,
)
_NO_ONE_ES = re.compile(r"\bnadie\s+(?:me\s+)?respond[iióo]\b", re.IGNORECASE)
_NO_ONE_EN = re.compile(
    r"\b(?:no one|nobody|no reply|no response|unanswered|"
    r"never replied|didn['’]t respond|did not respond)\b",
    re.IGNORECASE,
)


def fidelity_flags(source: str, translated: str) -> tuple[str, ...]:
    """Devuelve alertas acotadas; no infiere la polaridad global del texto."""
    flags: list[str] = []
    if _SEVERE_ES.search(source):
        # Se comprueba la primera oración, donde está la expresión severa
        # que Argos invirtió en la reproducción reportada.
        first_clause = re.split(r"[.!?]", translated, maxsplit=1)[0]
        if not _NEGATIVE_EN.search(first_clause):
            flags.append("severe_negative_expression_lost")
    if _NO_ONE_ES.search(source) and not _NO_ONE_EN.search(translated):
        flags.append("missing_no_response")
    return tuple(flags)


def safe_paraphrase(source: str) -> str:
    """Reformula únicamente expresiones del glosario controlado ES."""
    changes = (
        (r"\bp[eé]sima\s+experiencia\b", "una experiencia muy mala"),
        (r"\bp[eé]simo\s+servicio\b", "un servicio muy malo"),
        (r"\bmi\s+pedido\s+lleg[oó]\s+tarde\b", "mi orden llegó tarde"),
        (r"\bmis\s+reclamaciones\b", "mis quejas"),
    )
    result = source
    for pattern, replacement in changes:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def guarded_translate(source: str, translator: LocalTranslator) -> CheckedTranslation:
    """Traduce; ante indicadores críticos, reintenta con paráfrasis controlada.

    No clasifica si una inversión crítica persiste. Fuera de las expresiones
    cubiertas, la calidad de la traducción continúa siendo incierta.
    """
    translated = translator.translate(source)
    if not isinstance(translated, str) or not translated.strip():
        raise TranslationFidelityError("La traducción no produjo texto utilizable.")
    if not fidelity_flags(source, translated):
        return CheckedTranslation(text=translated)
    alternative = safe_paraphrase(source)
    if alternative == source:
        raise TranslationFidelityError(
            "La traducción podría cambiar el significado. Reformula el texto."
        )
    retried = translator.translate(alternative)
    if not isinstance(retried, str) or not retried.strip():
        raise TranslationFidelityError(
            "No se pudo verificar la traducción alternativa."
        )
    # La paráfrasis puede eliminar la expresión explícita, así que
    # exigimos que la traducción de la primera cláusula sea negativa.
    first_clause = re.split(r"[.!?]", retried, maxsplit=1)[0]
    if _SEVERE_ES.search(source) and not _NEGATIVE_EN.search(first_clause):
        raise TranslationFidelityError(
            "La traducción mantiene una posible inversión de sentimiento. "
            "Reformula el texto antes de clasificarlo."
        )
    if _NO_ONE_ES.search(source) and not _NO_ONE_EN.search(retried):
        raise TranslationFidelityError(
            "La traducción podría haber perdido una negación importante."
        )
    return CheckedTranslation(
        text=retried,
        quality_note=(
            "Se detectó una posible alteración de significado y se reintentó "
            "la traducción con una paráfrasis controlada. Revisa el inglés mostrado."
        ),
        paraphrase_used=True,
    )
