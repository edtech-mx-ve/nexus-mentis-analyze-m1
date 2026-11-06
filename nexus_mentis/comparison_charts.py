"""Gráficas comparativas reproducibles de validación para NEXUS ANALYZE.

No utiliza el conjunto de prueba para seleccionar ni comparar algoritmos.
Las figuras se producen a partir del report.json ya generado; no reentrenan modelos.
"""
from __future__ import annotations

import io
import logging
import math
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)

NAVY = "#183C52"
TEAL = "#158A7D"
GRAY = "#94AAB6"
INK = "#274B5C"
LIGHT = "#E5EDF0"
ChartType = Literal["model_quality.png", "training_time.png", "gain_vs_baseline.png"]
CHART_NAMES: frozenset[str] = frozenset(
    {"model_quality.png", "training_time.png", "gain_vs_baseline.png"}
)


def _number(value: Any, key: str, high: float | None = None) -> float:
    """Convierte y valida una métrica numérica finita y no negativa."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Valor inválido en {key}.")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (high is not None and number > high):
        raise ValueError(f"Fuera de rango: {key}.")
    return number


def _validate_report(report: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Lee línea base y resultados de validación del esquema actual."""
    if not isinstance(report, dict):
        raise ValueError("El reporte debe ser un diccionario.")
    baseline = report.get("baseline_validation")
    comparison = report.get("models_validation")
    winner = report.get("selected_model")
    if not isinstance(baseline, dict) or not isinstance(comparison, list) or not comparison:
        raise ValueError("Faltan métricas de validación para elaborar gráficos.")
    if not isinstance(winner, str) or not winner.strip():
        raise ValueError("No existe un modelo seleccionado.")
    if len(comparison) > 16:
        raise ValueError("Demasiados modelos para la visualización.")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [{"model": "Línea base (clase mayoritaria)", **baseline, "train_seconds": 0.0}, *comparison]:
        if not isinstance(row, dict):
            raise ValueError("Una fila del reporte es inválida.")
        name = row.get("model")
        if not isinstance(name, str) or not name.strip() or len(name) > 100 or name in seen:
            raise ValueError("Nombre de modelo inválido o duplicado.")
        seen.add(name)
        rows.append({
            "model": name,
            "f1": _number(row.get("f1_macro"), "f1_macro", high=1),
            "accuracy": _number(row.get("accuracy"), "accuracy", high=1),
            "seconds": _number(row.get("train_seconds", 0.0), "train_seconds"),
        })
    if winner not in seen:
        raise ValueError("El modelo seleccionado no aparece en validación.")
    if winner == "Línea base (clase mayoritaria)":
        raise ValueError("La línea base no puede ser el clasificador seleccionado.")
    return rows, winner


def _base_figure(height: float = 4.3) -> tuple[Any, Any]:
    """Crea un gráfico limpio con fondo y cuadrícula discretos."""
    fig, ax = plt.subplots(figsize=(9.8, height), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(axis="both", colors=INK, labelsize=10, length=0, pad=8)
    ax.grid(axis="x", color=LIGHT, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def _quality(rows: list[dict[str, Any]], winner: str) -> Any:
    """Compara F1 macro y Accuracy en una escala honesta de 0 a 100 %."""
    fig, ax = _base_figure(4.9)
    positions = np.arange(len(rows))
    f1 = np.array([row["f1"] * 100 for row in rows])
    accuracy = np.array([row["accuracy"] * 100 for row in rows])
    labels = [str(row["model"]) for row in rows]
    width = 0.32
    ax.barh(positions - width / 2, f1, height=width, color=TEAL,
            label="F1 macro", zorder=3)
    ax.barh(positions + width / 2, accuracy, height=width, color=NAVY,
            label="Accuracy", zorder=3)
    for i in range(len(rows)):
        ax.text(f1[i] + 1.2, i - width / 2, f"{f1[i]:.1f} %",
                va="center", ha="left", fontsize=9, color=INK)
        ax.text(accuracy[i] + 1.2, i + width / 2, f"{accuracy[i]:.1f} %",
                va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 114)  # datos 0-100; margen a la derecha solo para etiquetas
    ax.set_xticks([0, 20, 40, 60, 80, 100], [f"{v} %" for v in (0, 20, 40, 60, 80, 100)])
    ax.set_title("Calidad de clasificación en validación", loc="left",
                 fontsize=14, weight="bold", color=NAVY, pad=19)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.84, bottom=0.12)
    return fig


def _training(rows: list[dict[str, Any]], winner: str) -> Any:
    """Muestra tiempos medidos de entrenamiento, sin inferir velocidad de API."""
    fig, ax = _base_figure(4.2)
    actual = rows[1:]  # la línea base no entrena un clasificador de texto
    y = np.arange(len(actual))
    values = [row["seconds"] for row in actual]
    highest = max(values, default=0.0)
    colors = [TEAL if row["model"] == winner else GRAY for row in actual]
    ax.barh(y, values, height=0.55, color=colors, zorder=3)
    for i, value in enumerate(values):
        ax.text(value + max(0.05, highest * 0.02), i,
                f"{value:.3f} s" if value >= 0.001 else "< 0,001 s",
                va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [str(row["model"]) for row in actual])
    ax.invert_yaxis()
    ax.set_xlim(0, highest * 1.25 if highest > 0 else 1)
    ax.set_xlabel("Segundos (menor = entrenamiento más rápido)", color=INK, fontsize=10)
    ax.set_title("Tiempo de entrenamiento por algoritmo", loc="left",
                 fontsize=14, weight="bold", color=NAVY, pad=17)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.82, bottom=0.21)
    return fig


def _baseline_gain(rows: list[dict[str, Any]], winner: str) -> Any:
    """Contrasta F1 macro de los modelos frente a la referencia trivial."""
    fig, ax = _base_figure(4.2)
    base = rows[0]["f1"] * 100
    actual = rows[1:]
    y = np.arange(len(actual))
    gains = [(row["f1"] * 100) - base for row in actual]
    lo, hi = min([0.0, *gains]), max([0.0, *gains])
    span = max(5, hi - lo)
    colors = [TEAL if row["model"] == winner else GRAY for row in actual]
    ax.barh(y, gains, height=0.55, color=colors, zorder=3)
    for i, gain in enumerate(gains):
        ax.text(gain + (0.02 * span if gain >= 0 else -0.02 * span), i,
                f"{gain:+.2f} pp", va="center",
                ha="left" if gain >= 0 else "right", fontsize=10, color=INK)
    ax.set_yticks(y, [str(row["model"]) for row in actual])
    ax.invert_yaxis()
    ax.set_xlim(lo - 0.15 * span, hi + 0.25 * span)
    ax.axvline(0, color=NAVY, linewidth=1)
    ax.set_xlabel("Diferencia de F1 macro (puntos porcentuales)", color=INK, fontsize=10)
    ax.set_title("Mejora respecto a la línea base", loc="left",
                 fontsize=14, weight="bold", color=NAVY, pad=17)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.82, bottom=0.21)
    return fig


def render_comparison_chart(report: dict[str, Any], chart_name: str) -> bytes:
    """Renderiza el PNG solicitado a partir de métricas reales del reporte.

    Args:
        report: Diccionario de report.json cargado localmente.
        chart_name: Nombre permitido de la figura.
    Returns:
        Bytes PNG sin crear archivos ni reentrenar.
    Raises:
        ValueError: Reporte o nombre de figura no válido.
    """
    if chart_name not in CHART_NAMES:
        raise ValueError(f"Gráfico no permitido: {chart_name}")
    rows, winner = _validate_report(report)
    figure = None
    try:
        if chart_name == "model_quality.png":
            figure = _quality(rows, winner)
        elif chart_name == "training_time.png":
            figure = _training(rows, winner)
        else:
            figure = _baseline_gain(rows, winner)
        with io.BytesIO() as output:
            figure.savefig(output, format="png", dpi=145, facecolor="white")
            return output.getvalue()
    finally:
        if figure is not None:
            plt.close(figure)
