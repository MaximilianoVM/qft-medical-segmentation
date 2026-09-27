"""
Archivo de pruebas unitarias: test_evaluation.py
Ubicación en el repositorio: tests/test_evaluation.py

Valida la reconstrucción de máscara a partir de predicciones por
superpíxel, y el cálculo de IoU, Dice, precisión y recall.
"""

import numpy as np
import pytest
from src.evaluation import (
    reconstruir_mascara,
    calcular_iou,
    calcular_dice,
    calcular_metricas,
)


# ---------------------------------------------------------------------------
# reconstruir_mascara
# ---------------------------------------------------------------------------

def test_reconstruir_mascara_forma_y_valores():
    region_izq = np.zeros((10, 10), dtype=bool)
    region_izq[:, :5] = True
    region_der = np.zeros((10, 10), dtype=bool)
    region_der[:, 5:] = True

    y_pred = np.array([1, 0])
    mascara = reconstruir_mascara([region_izq, region_der], y_pred, forma=(10, 10))

    assert mascara.shape == (10, 10)
    assert np.all(mascara[:, :5] == 1)
    assert np.all(mascara[:, 5:] == 0)


def test_reconstruir_mascara_longitudes_distintas():
    region = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        reconstruir_mascara([region], np.array([1, 0]), forma=(5, 5))


# ---------------------------------------------------------------------------
# calcular_iou
# ---------------------------------------------------------------------------

def test_calcular_iou_identicas():
    mascara = np.zeros((10, 10), dtype=int)
    mascara[2:6, 2:6] = 1
    assert calcular_iou(mascara, mascara) == pytest.approx(1.0, abs=1e-4)


def test_calcular_iou_sin_traslape():
    pred = np.zeros((10, 10), dtype=int)
    pred[0:3, 0:3] = 1
    real = np.zeros((10, 10), dtype=int)
    real[7:10, 7:10] = 1
    assert calcular_iou(pred, real) == pytest.approx(0.0, abs=1e-4)


def test_calcular_iou_traslape_parcial():
    pred = np.zeros((10, 10), dtype=int)
    pred[0:4, 0:4] = 1  # 16 píxeles
    real = np.zeros((10, 10), dtype=int)
    real[2:6, 2:6] = 1  # 16 píxeles, traslape en [2:4, 2:4] = 4 píxeles

    # interseccion=4, union=16+16-4=28
    assert calcular_iou(pred, real) == pytest.approx(4 / 28, abs=1e-4)


def test_calcular_iou_ambas_vacias():
    vacia = np.zeros((10, 10), dtype=int)
    assert calcular_iou(vacia, vacia) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# calcular_dice
# ---------------------------------------------------------------------------

def test_calcular_dice_identicas():
    mascara = np.zeros((10, 10), dtype=int)
    mascara[2:6, 2:6] = 1
    assert calcular_dice(mascara, mascara) == pytest.approx(1.0, abs=1e-4)


def test_calcular_dice_sin_traslape():
    pred = np.zeros((10, 10), dtype=int)
    pred[0:3, 0:3] = 1
    real = np.zeros((10, 10), dtype=int)
    real[7:10, 7:10] = 1
    assert calcular_dice(pred, real) == pytest.approx(0.0, abs=1e-4)


def test_calcular_dice_traslape_parcial():
    pred = np.zeros((10, 10), dtype=int)
    pred[0:4, 0:4] = 1  # 16 píxeles
    real = np.zeros((10, 10), dtype=int)
    real[2:6, 2:6] = 1  # 16 píxeles, interseccion=4

    # dice = 2*4 / (16+16) = 8/32
    assert calcular_dice(pred, real) == pytest.approx(8 / 32, abs=1e-4)


def test_calcular_dice_ambas_vacias():
    vacia = np.zeros((10, 10), dtype=int)
    assert calcular_dice(vacia, vacia) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# calcular_metricas
# ---------------------------------------------------------------------------

def test_calcular_metricas_claves_esperadas():
    mascara = np.zeros((10, 10), dtype=int)
    mascara[2:6, 2:6] = 1
    metricas = calcular_metricas(mascara, mascara)

    assert set(metricas.keys()) == {"iou", "dice", "precision", "recall"}


def test_calcular_metricas_prediccion_perfecta():
    mascara = np.zeros((10, 10), dtype=int)
    mascara[2:6, 2:6] = 1
    metricas = calcular_metricas(mascara, mascara)

    assert metricas["iou"] == pytest.approx(1.0, abs=1e-4)
    assert metricas["dice"] == pytest.approx(1.0, abs=1e-4)
    assert metricas["precision"] == pytest.approx(1.0, abs=1e-4)
    assert metricas["recall"] == pytest.approx(1.0, abs=1e-4)


def test_calcular_metricas_falsos_positivos():
    # Predicción cubre TODO como lesión; el real solo tiene una parte.
    pred = np.ones((10, 10), dtype=int)
    real = np.zeros((10, 10), dtype=int)
    real[2:6, 2:6] = 1  # 16 píxeles de 100

    metricas = calcular_metricas(pred, real)

    assert metricas["recall"] == pytest.approx(1.0, abs=1e-4)   # detectó toda la lesión real
    assert metricas["precision"] == pytest.approx(16 / 100, abs=1e-4)  # pero con muchos falsos positivos


def test_calcular_metricas_consistencia_iou_dice():
    """Dice y IoU están relacionados: Dice = 2*IoU / (1+IoU).
        
        Sirve como chequeo cruzado de que ambas fórmulas están bien implementadas, no solo que cada una da el número "esperado" por separado.
    """
    pred = np.zeros((10, 10), dtype=int)
    pred[0:4, 0:4] = 1
    real = np.zeros((10, 10), dtype=int)
    real[2:6, 2:6] = 1

    metricas = calcular_metricas(pred, real)
    iou = metricas["iou"]
    dice_esperado = 2 * iou / (1 + iou)

    assert metricas["dice"] == pytest.approx(dice_esperado, abs=1e-3)