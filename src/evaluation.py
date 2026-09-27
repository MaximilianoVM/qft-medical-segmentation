"""
Módulo: evaluation.py
Implementación de los Pasos 9 y 10: Reconstrucción de máscara y evaluación
(IoU, Dice, precisión, recall) contra el ground-truth.
"""

import numpy as np


def reconstruir_mascara(regiones_sp: list[np.ndarray], y_pred: np.ndarray, forma: tuple[int, int]) -> np.ndarray:
    """
    Paso 9: Reconstruye una máscara binaria (H, W) a partir de las
    predicciones por superpíxel: cada píxel toma la clase predicha
    de su superpíxel.

    Parámetros:
        regiones_sp (list[np.ndarray]): lista de K máscaras booleanas (H, W),
            en el mismo orden en que se generó y_pred.
        y_pred (np.ndarray): vector de K etiquetas predichas (0/1).
        forma (tuple[int, int]): dimensiones (H, W) de la imagen original.

    Retorna:
        np.ndarray: máscara binaria (H, W) de tipo int.
    """
    if len(regiones_sp) != len(y_pred):
        raise ValueError(
            f"regiones_sp ({len(regiones_sp)}) y y_pred ({len(y_pred)}) "
            f"deben tener la misma longitud y estar en el mismo orden."
        )

    mascara_pred = np.zeros(forma, dtype=int)
    for region, etiqueta in zip(regiones_sp, y_pred):
        mascara_pred[region] = etiqueta

    return mascara_pred


def calcular_iou(mascara_pred: np.ndarray, mascara_real: np.ndarray, eps: float = 1e-8) -> float:
    """
    Intersection over Union entre la máscara predicha y el ground-truth.
    IoU = |pred ∩ real| / |pred ∪ real|
    """
    pred_bool = mascara_pred > 0
    real_bool = mascara_real > 0

    interseccion = np.logical_and(pred_bool, real_bool).sum()
    union = np.logical_or(pred_bool, real_bool).sum()

    if union == 0:
        return 1.0  # ambas máscaras vacías: perfecto acuerdo por convención

    return float(interseccion / (union + eps))


def calcular_dice(mascara_pred: np.ndarray, mascara_real: np.ndarray, eps: float = 1e-8) -> float:
    """
    Coeficiente Dice (F1 a nivel de píxel).
    Dice = 2*|pred ∩ real| / (|pred| + |real|)
    """
    pred_bool = mascara_pred > 0
    real_bool = mascara_real > 0

    interseccion = np.logical_and(pred_bool, real_bool).sum()
    suma = pred_bool.sum() + real_bool.sum()

    if suma == 0:
        return 1.0  # ambas máscaras vacías: perfecto acuerdo por convención

    return float(2 * interseccion / (suma + eps))


def calcular_metricas(mascara_pred: np.ndarray, mascara_real: np.ndarray, eps: float = 1e-8) -> dict:
    """
    Calcula IoU, Dice, precisión y recall a nivel de píxel en un solo dict.

    Precisión = |pred ∩ real| / |pred|   (de lo que predije como lesión, cuánto era real)
    Recall    = |pred ∩ real| / |real|   (de la lesión real, cuánto detecté)
    """
    pred_bool = mascara_pred > 0
    real_bool = mascara_real > 0

    interseccion = np.logical_and(pred_bool, real_bool).sum()
    total_pred = pred_bool.sum()
    total_real = real_bool.sum()

    precision = float(interseccion / (total_pred + eps)) if total_pred > 0 else (1.0 if total_real == 0 else 0.0)
    recall = float(interseccion / (total_real + eps)) if total_real > 0 else (1.0 if total_pred == 0 else 0.0)

    return {
        "iou": calcular_iou(mascara_pred, mascara_real, eps=eps),
        "dice": calcular_dice(mascara_pred, mascara_real, eps=eps),
        "precision": precision,
        "recall": recall,
    }