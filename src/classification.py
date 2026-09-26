"""
Módulo: classification.py
Implementación de los Pasos 6, 7 y 8: Asignación de clase (etiquetado),
entrenamiento del clasificador y predicción sobre vectores de características.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def etiquetar_region(mascara_region: np.ndarray, umbral: float = 0.5) -> int:
    """
    Paso 6: Asigna clase 0/1 a una región según la fracción de sus píxeles
    que pertenecen a la clase positiva (lesión) en la máscara ground-truth.

    Parámetros:
        mascara_region (np.ndarray): valores de la máscara ground-truth
            correspondientes a los píxeles de la región (p. ej. mascara[region]
            si region es una máscara booleana, o mascara[filas, columnas]).
        umbral (float): fracción mínima de píxeles positivos para asignar
            clase 1 (lesión). Por defecto 0.5.

    Retorna:
        int: 0 (piel sana) o 1 (lesión).
    """
    if mascara_region.size == 0:
        return 0
    fraccion = np.sum(mascara_region > 0) / mascara_region.size
    return 1 if fraccion >= umbral else 0


def etiquetar_superpixeles(regiones_sp: list[np.ndarray], mascara: np.ndarray, umbral: float = 0.5) -> np.ndarray:
    """
    Paso 6: Aplica etiquetar_region a cada superpíxel de una imagen.

    Parámetros:
        regiones_sp (list[np.ndarray]): lista de K máscaras booleanas (H, W),
            una por superpíxel.
        mascara (np.ndarray): máscara ground-truth completa (H, W).
        umbral (float): fracción mínima de traslape con la lesión para
            etiquetar como clase 1.

    Retorna:
        np.ndarray: vector de K etiquetas (0/1), una por superpíxel.
    """
    return np.array([
        etiquetar_region(mascara[region], umbral=umbral)
        for region in regiones_sp
    ])


_MODELOS_DISPONIBLES = {
    "svm": lambda class_weight=None: SVC(kernel="rbf", random_state=42, class_weight=class_weight),
    "knn": lambda class_weight=None: KNeighborsClassifier(n_neighbors=5),  # KNN no soporta class_weight
    "random_forest": lambda class_weight=None: RandomForestClassifier(random_state=42, class_weight=class_weight),
}


def entrenar_modelo(X: np.ndarray, y: np.ndarray, tipo_modelo: str = "svm", class_weight: str | None = None):
    """
    Paso 7: Entrena un clasificador sobre la matriz de características X
    y las etiquetas y, y lo devuelve ya entrenado.

    Parámetros:
        X (np.ndarray): matriz de características (K x F).
        y (np.ndarray): vector de etiquetas (K,).
        tipo_modelo (str): "svm", "knn" o "random_forest".
        class_weight (str | None): "balanced" para compensar clases
            desbalanceadas (no soportado por knn, se ignora si se pasa).

    Retorna:
        Clasificador de scikit-learn ya entrenado (método .fit ya llamado).
    """
    if tipo_modelo not in _MODELOS_DISPONIBLES:
        raise ValueError(
            f"tipo_modelo '{tipo_modelo}' no soportado. "
            f"Usar uno de: {list(_MODELOS_DISPONIBLES.keys())}."
        )

    modelo = _MODELOS_DISPONIBLES[tipo_modelo](class_weight=class_weight) 
    modelo.fit(X, y)
    return modelo


def predecir(modelo, X: np.ndarray) -> np.ndarray:
    """
    Paso 8: Devuelve las etiquetas predichas para un conjunto de vectores
    de características, usando un modelo ya entrenado.

    Parámetros:
        modelo: clasificador de scikit-learn entrenado (salida de entrenar_modelo).
        X (np.ndarray): matriz de características (K x F).

    Retorna:
        np.ndarray: vector de K etiquetas predichas (0/1).
    """
    return modelo.predict(X)