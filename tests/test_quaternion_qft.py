"""
Archivo de pruebas unitarias: test_quaternion_qft.py
Ubicación en el repositorio: tests/test_quaternion_qft.py

Valida la conversión a cuaterniones y el cálculo de la QFT 2D por superpíxel.
"""

import numpy as np
import pytest
from src.quaternion_qft import (
    rgb_a_cuaternion,
    qft_2d_region,
    procesar_qft_superpixeles,
)


def test_rgb_a_cuaternion_dimensiones_y_rangos():
    img_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    q_arr = rgb_a_cuaternion(img_uint8)

    assert q_arr.shape == (50, 50, 4)
    assert np.all(q_arr[:, :, 0] == 0.0)  # Parte real nula
    assert q_arr[:, :, 1:].max() <= 1.0   # Valores normalizados entre 0 y 1
    assert q_arr[:, :, 1:].min() >= 0.0


def test_rgb_a_cuaternion_validacion_tipo():
    with pytest.raises(TypeError):
        rgb_a_cuaternion([[1, 2, 3]])

    with pytest.raises(ValueError):
        rgb_a_cuaternion(np.zeros((50, 50)))  # No es 3D


def test_qft_2d_region_dimensiones():
    q_sub = np.random.randn(20, 20, 4)
    spectrum = qft_2d_region(q_sub)

    assert spectrum.shape == (20, 20, 4)
    assert np.iscomplexobj(spectrum)


def test_procesar_qft_superpixeles_integracion():
    img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
    mask1 = np.zeros((40, 40), dtype=bool)
    mask1[:20, :] = True
    mask2 = np.zeros((40, 40), dtype=bool)
    mask2[20:, :] = True

    espectros = procesar_qft_superpixeles(img, [mask1, mask2])

    assert len(espectros) == 2
    assert espectros[0].shape == (20, 40, 4)
    assert espectros[1].shape == (20, 40, 4)
