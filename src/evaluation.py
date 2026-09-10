import numpy as np

def iou(mascara_pred: np.ndarray, mascara_real: np.ndarray) -> float:
    """Calcula Intersection over Union entre la máscara predicha y la real."""
    pass

def dice(mascara_pred: np.ndarray, mascara_real: np.ndarray) -> float:
    """Calcula el coeficiente Dice."""
    pass

def precision_recall(mascara_pred: np.ndarray, mascara_real: np.ndarray) -> tuple[float, float]:
    """Calcula precisión y sensibilidad (recall)."""
    pass