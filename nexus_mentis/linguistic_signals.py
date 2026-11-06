"""Señales lingüísticas de riesgo para inferencia bilingüe.

Reglas heurísticas explicables. No constituyen un detector validado de sarcasmo
ni modifican las etiquetas producidas por el clasificador entrenado.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


_EMOJI_PATTERN = (
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"
    r"(?:[\ufe0e\ufe0f]|[\U0001F3FB-\U0001F3FF])*"
    r"(?:\u200d[\U0001F300-\U0001FAFF\u2600-\u27BF][\ufe0e\ufe0f]*)*"
)
_EMOJI = re.compile(_EMOJI_PATTERN)
# Solo símbolos *sueltos al final*, antes o después de emojis;
# no se elimina puntuación dentro de oraciones ni caracteres del original.
_TRAILING_ARTIFACT = re.compile(
    rf"(?P<junk>\s+(?P<symbol>[Οο♪♫]|[-–—]))"
    rf"(?P<tail>(?:\s+{_EMOJI_PATTERN})*)\s*$"
)
_EYEROLL = {"🙄", "😒", "😑", "😤"}
_ES_POSITIVE_IRONIC = re.compile(
    r"(?:qu[eé]\s+maravilla|qu[eé]\s+genial|qu[eé]\s+bien|"
    r"qu[eé]\s+(?:excelente|fant[aá]stico|maravilloso)\s+"
    r"(?:servicio|atenci[oó]n)|"
    r"gracias\s+por\s+nada|s[ií][, ]+claro|fant[aá]stico)",
    re.IGNORECASE,
)
_ES_NEGATIVE_CONTEXT = re.compile(
    r"(?:horas?\s+esperando|tres\s+horas|retras|tard[eó]|"
    r"p[eé]sim|horribl|terribl|nadie|quejas?|"
    r"no\s+(?:funcion|lleg|respondi|sirv)|"
    r"esperando\s+desde|incompleto)",
    re.IGNORECASE,
)
_EN_IRONIC = re.compile(
    r"(?:how\s+wonderful|just\s+great|thanks\s+for\s+nothing|"
    r"oh\s+great|yeah\s+right|fantastic)",
    re.IGNORECASE,
)
_EN_NEGATIVE_CONTEXT = re.compile(
    r"(?:hours?\s+waiting|late|delay|terrible|awful|"
    r"nobody|no\s+one|broken|incomplete|waiting\s+for\s+hours?)",
    re.IGNORECASE,
)
_ES_NEGATION = re.compile(r"\b(?:no|nunca|jam[aá]s|nadie|sin)\b", re.IGNORECASE)
_EN_NEGATION = re.compile(
    r"\b(?:not|no|never|nobody|nothing|neither|without|cannot|can't|"
    r"didn't|don't|doesn't|isn't|wasn't|won't|hardly)\b", re.IGNORECASE,
)
_ES_MIXED = re.compile(
    r"\b(?:aunque|pero|sin embargo|a pesar de)\b", re.IGNORECASE
)
_ES_POSITIVE_AFTER_CONTRAST = re.compile(
    r"\b(?:pero|sin embargo)\b[^.!?]{0,150}"
    r"\b(?:fant[aá]stic[oa]s?|excelente(?:s)?|incre[ií]ble(?:s)?|"
    r"genial(?:es)?|maravillos[oa]s?|me encant[oó]|estupend[oa]s?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LinguisticAssessment:
    """Observaciones trazables sin sustituir la decisión del modelo."""
    translated_text: str
    flags: tuple[str, ...]
    notes: tuple[str, ...]
    source_emojis: tuple[str, ...]
    emojis_restored: bool

    @property
    def interpretation_limited(self) -> bool:
        return bool(self.flags)


def extract_emojis(text: str) -> tuple[str, ...]:
    """Extrae emojis en orden; no modifica el texto original."""
    if not isinstance(text, str):
        raise TypeError("El texto debe ser una cadena.")
    return tuple(match.group() for match in _EMOJI.finditer(text))


def strip_translation_artifacts(original: str, translated: str) -> tuple[str, tuple[str, ...]]:
    """Quita exclusivamente marcadores finales aislados introducidos por Argos.

    Se activa solo con emoji en el original. No corrige gramática, sarcasmo,
    puntuación interna ni expresiones de origen: limpieza conservadora.
    """
    if not isinstance(original, str) or not isinstance(translated, str):
        raise TypeError("Se requieren cadenas.")
    if not extract_emojis(original):
        return translated, ()
    cleaned = translated
    removed: list[str] = []
    while True:
        match = _TRAILING_ARTIFACT.search(cleaned)
        if match is None:
            break
        symbol = match.group("symbol")
        if symbol in original:
            break  # posible símbolo intencional del autor original
        cleaned = cleaned[:match.start("junk")] + match.group("tail")
        removed.append(symbol)
    return cleaned, tuple(removed)


def restore_emojis(original: str, translated: str) -> tuple[str, bool]:
    """Conserva emojis eliminados por el traductor, sin duplicarlos.

    Se anexan al final si faltan porque la realineación palabra a palabra
    no puede garantizarse. El modelo sigue siendo el mismo, sin recalibrar.
    """
    if not isinstance(original, str) or not isinstance(translated, str):
        raise TypeError("Se requieren cadenas.")
    existing = Counter(extract_emojis(translated))
    missing: list[str] = []
    for emoji in extract_emojis(original):
        if existing[emoji]:
            existing[emoji] -= 1
        else:
            missing.append(emoji)
    if not missing:
        return translated, False
    return translated.rstrip() + " " + " ".join(missing), True


def inspect_signals(original: str, translated: str, language: str) -> LinguisticAssessment:
    """Devuelve alertas acotadas para español o inglés sin reclasificar.

    La preservación de emojis es explícita; 'sarcasmo' solo es sospecha y
    no se presenta como detección semántica fiable.
    """
    if language not in {"es", "en"}:
        raise ValueError("Idioma inválido.")
    if not isinstance(original, str) or not isinstance(translated, str):
        raise TypeError("Los textos deben ser cadenas.")

    cleaned, artifacts_removed = (
        strip_translation_artifacts(original, translated) if language == "es"
        else (translated, ())
    )
    result, restored = restore_emojis(original, cleaned) if language == "es" \
        else (translated, False)
    emojis = extract_emojis(original)
    flags: list[str] = []
    notes: list[str] = []
    if artifacts_removed:
        flags.append("translation_artifact_removed")
        notes.append(
            "Se quitaron símbolos aislados al final de la traducción que no "
            "figuraban en el original y aparecían junto a emojis."
        )
    if restored:
        flags.append("emoji_restored")
        notes.append(
            "El traductor eliminó uno o más emojis; se conservaron en el "
            "texto inglés al final, sin interpretar automáticamente su sentimiento."
        )
    if any(emoji in _EYEROLL for emoji in emojis):
        flags.append("expressive_irony_cue")
        notes.append(
            "Hay un emoji asociado a ironía o desaprobación; la predicción "
            "literal puede no representar la intención del mensaje."
        )
    if language == "es":
        likely_ironic = bool(_ES_POSITIVE_IRONIC.search(original) and
                             _ES_NEGATIVE_CONTEXT.search(original))
        if _ES_NEGATION.search(original) and not _EN_NEGATION.search(result):
            flags.append("possible_negation_loss")
            notes.append(
                "Posible pérdida de una negación durante la traducción; "
                "comprueba el texto inglés antes de interpretar la predicción."
            )
        if _ES_MIXED.search(original):
            flags.append("mixed_statement")
            notes.append(
                "El texto contiene un conector de contraste; puede expresar "
                "aspectos positivos y negativos a la vez."
            )
            if _ES_POSITIVE_AFTER_CONTRAST.search(original):
                flags.append("positive_final_clause")
                notes.append(
                    "La cláusula posterior al contraste contiene una valoración "
                    "positiva. La predicción puede no reflejar ese cierre; "
                    "se conserva la etiqueta original del modelo."
                )
    else:
        likely_ironic = bool(_EN_IRONIC.search(original) and
                             _EN_NEGATIVE_CONTEXT.search(original))
    if likely_ironic:
        flags.append("possible_sarcasm")
        notes.append(
            "Se detectó una combinación de elogio literal y contexto "
            "desfavorable: posible sarcasmo. El modelo no está entrenado "
            "específicamente para resolverlo."
        )
    return LinguisticAssessment(
        translated_text=result,
        flags=tuple(dict.fromkeys(flags)),
        notes=tuple(notes),
        source_emojis=emojis,
        emojis_restored=restored,
    )
