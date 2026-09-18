import numpy as np

#def extraer_caracteristicas(espectro_qft: np.ndarray) -> np.ndarray:
    # Calcula el vector de características (magnitud, fase, energía, entropía, etc.) a partir del espectro QFT.
 #   pass

#def normalizar_caracteristicas(vectores: np.ndarray) -> np.ndarray:
    # Normaliza los vectores de características (ej. StandardScaler o Min-Max).
#    pass

"""
Módulo: feature_extraction.py
Implementación de los Pasos 4, 5 y 6: Extracción y Normalización de Características Espectrales QFT.
"""

import numpy as np


def calcular_magnitud_espectral(F_k: np.ndarray) -> np.ndarray:
    """
    Calcula la magnitud cuaterniónica |F(u,v)| de un espectro QFT 2D.
    Entrada: F_k array de forma (M, N, 4)
    Salida: Magnitud 2D de forma (M, N)
    """
    if not isinstance(F_k, np.ndarray) or F_k.ndim != 3 or F_k.shape[2] != 4:
        raise ValueError(f"F_k debe ser un np.ndarray de forma (M, N, 4). Recibido: {type(F_k)}")
    
    # Magnitud cuaterniónica: sqrt(|F_w|^2 + |F_x|^2 + |F_y|^2 + |F_z|^2)
    magnitud_cuadratica = np.sum(np.abs(F_k) ** 2, axis=-1)
    return np.sqrt(magnitud_cuadratica)


def calcular_energia_espectral(F_k: np.ndarray) -> float:
    """
    Suma de las magnitudes al cuadrado normalizada por el número de píxeles de la región.
    """
    mag = calcular_magnitud_espectral(F_k)
    return float(np.sum(mag ** 2) / mag.size)


def calcular_entropia_espectral(F_k: np.ndarray, eps: float = 1e-12) -> float:
    """
    Entropía de Shannon sobre la distribución normalizada de energía espectral.
    """
    mag = calcular_magnitud_espectral(F_k)
    p_uv = (mag ** 2) / (np.sum(mag ** 2) + eps)
    p_uv = p_uv[p_uv > eps]  # Evitar log2(0)
    return float(-np.sum(p_uv * np.log2(p_uv)))


def extraer_vector_caracteristicas_region(F_k: np.ndarray) -> np.ndarray:
    """
    Extrae un vector 1D de características espectrales a partir del espectro QFT F_k.
    Vector F = [media_mag, std_mag, max_mag, energia, entropia]
    """
    mag = calcular_magnitud_espectral(F_k)
    
    media_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))
    max_mag = float(np.max(mag))
    energia = calcular_energia_espectral(F_k)
    entropia = calcular_entropia_espectral(F_k)
    
    return np.array([media_mag, std_mag, max_mag, energia, entropia], dtype=np.float64)


def extraer_matriz_caracteristicas(espectros_qft: list[np.ndarray]) -> np.ndarray:
    """
    Genera la matriz X de dimensión (K x F) para los K superpíxeles de una imagen.
    """
    if not espectros_qft:
        return np.empty((0, 5), dtype=np.float64)
    
    vectores = [extraer_vector_caracteristicas_region(F_k) for F_k in espectros_qft]
    return np.vstack(vectores)


def normalizar_caracteristicas(X: np.ndarray, metodo: str = "zscore", eps: float = 1e-8) -> np.ndarray:
    """
    Normaliza la matriz de características X (K x F) usando Z-score o MinMax.
    (Paso 6 de la metodología).
    """
    if X.size == 0:
        return X
    
    if metodo == "zscore":
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        return (X - mu) / (sigma + eps)
    elif metodo == "minmax":
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        return (X - x_min) / (x_max - x_min + eps)
    else:
        raise ValueError(f"Método no soportado: {metodo}. Usar 'zscore' o 'minmax'.")
