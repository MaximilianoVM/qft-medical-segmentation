"""
Pruebas unitarias para el módulo feature_extraction.py (Pasos 4, 5 y 6)
"""

import numpy as np
from src.feature_extraction import (
    calcular_magnitud_espectral,
    calcular_energia_espectral,
    calcular_entropia_espectral,
    extraer_vector_caracteristicas_region,
    extraer_matriz_caracteristicas,
    normalizar_caracteristicas,
)


def test_calcular_magnitud_espectral():
    F_k = np.ones((10, 10, 4), dtype=np.complex128)
    mag = calcular_magnitud_espectral(F_k)
    assert mag.shape == (10, 10)
    assert np.allclose(mag, 2.0)


def test_energia_y_entropia():
    F_k = np.ones((5, 5, 4), dtype=np.complex128)
    energia = calcular_energia_espectral(F_k)
    entropia = calcular_entropia_espectral(F_k)
    assert isinstance(energia, float)
    assert isinstance(entropia, float)
    assert energia > 0
    assert entropia >= 0


def test_extraer_matriz_caracteristicas():
    espectros = [np.ones((8, 8, 4), dtype=np.complex128) for _ in range(3)]
    X = extraer_matriz_caracteristicas(espectros)
    assert X.shape == (3, 5)


def test_normalizar_caracteristicas_zscore():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    X_norm = normalizar_caracteristicas(X, metodo="zscore")
    assert X_norm.shape == X.shape
    assert np.allclose(np.mean(X_norm, axis=0), 0.0, atol=1e-7)


def test_normalizar_caracteristicas_minmax():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    X_norm = normalizar_caracteristicas(X, metodo="minmax")
    assert X_norm.shape == X.shape
    assert np.allclose(np.min(X_norm, axis=0), 0.0, atol=1e-7)
    assert np.allclose(np.max(X_norm, axis=0), 1.0, atol=1e-7)
