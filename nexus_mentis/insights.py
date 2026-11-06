"""Interpretación didáctica y determinista de métricas de clasificación.

No incorpora observaciones del dataset ni estima significación estadística.
Todas las conclusiones proceden exclusivamente de métricas recibidas.
"""
from __future__ import annotations

from typing import Any


def summarize_selection(
    comparison: list[dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Resume el resultado de validación sin confundirlo con prueba estadística."""
    if not comparison:
        raise ValueError("Se requiere al menos un modelo evaluado.")
    ordered = sorted(
        comparison,
        key=lambda item: (float(item["f1_macro"]), float(item["accuracy"])),
        reverse=True,
    )
    winner = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    gap = 100.0 * (float(winner["f1_macro"]) - float(second["f1_macro"])) if second else None
    baseline_gap = 100.0 * (
        float(winner["f1_macro"]) - float(baseline["f1_macro"])
    )
    return {
        "winner": str(winner["model"]),
        "runner_up": str(second["model"]) if second else None,
        "gap_points": round(gap, 2) if gap is not None else None,
        "baseline_gap_points": round(baseline_gap, 2),
        "caution": (
            "La selección se hizo por F1 macro en validación. La diferencia no "
            "prueba superioridad universal ni significación estadística."
        ),
    }


def summarize_confusion(matrix: list[list[int]]) -> dict[str, Any]:
    """Deriva aciertos, errores y tasas por clase de una matriz binaria."""
    if (
        len(matrix) != 2 or any(len(row) != 2 for row in matrix)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for row in matrix for value in row)
    ):
        raise ValueError("Se requiere una matriz 2x2 de enteros no negativos.")
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    if total < 1 or tn + fp < 1 or fn + tp < 1:
        raise ValueError("La matriz debe contener ejemplos de ambas clases.")
    correct = tn + tp
    errors = fp + fn
    return {
        "total": total,
        "correct": correct,
        "errors": errors,
        "error_pct": round(100 * errors / total, 2),
        "negative_as_positive": fp,
        "positive_as_negative": fn,
        "negative_recall": round(tn / (tn + fp), 6),
        "positive_recall": round(tp / (tp + fn), 6),
    }
