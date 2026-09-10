import numpy as np

def etiquetar_region(mascara_region: np.ndarray) -> int:
    """Asigna una clase a la región según la clase mayoritaria en la máscara de referencia."""
    pass

def entrenar_modelo(X: np.ndarray, y: np.ndarray, tipo_modelo: str = "svm"):
    """Entrena un clasificador (svm, knn, random_forest) y lo devuelve entrenado."""
    pass

def predecir(modelo, X: np.ndarray) -> np.ndarray:
    """Devuelve las etiquetas predichas para un conjunto de vectores de características."""
    pass