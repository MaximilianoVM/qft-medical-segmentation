"""
Archivo de pruebas unitarias: test_classification.py
Ubicación en el repositorio: tests/test_classification.py

Valida el etiquetado por región/superpíxel, el entrenamiento de
clasificadores (SVM, KNN, Random Forest) y la predicción sobre
vectores de características.
"""

import numpy as np
import pytest
from src.classification import (
    etiquetar_region,
    etiquetar_superpixeles,
    entrenar_modelo,
    predecir,
)


# ---------------------------------------------------------------------------
# etiquetar_region
# ---------------------------------------------------------------------------

def test_etiquetar_region_mayoria_lesion():
    mascara_region = np.array([1, 1, 1, 0, 0], dtype=int)  # 60% clase 1
    assert etiquetar_region(mascara_region) == 1  # umbral por defecto 0.5


def test_etiquetar_region_mayoria_sana():
    mascara_region = np.array([0, 0, 0, 1, 1], dtype=int)  # 60% clase 0
    assert etiquetar_region(mascara_region) == 0


def test_etiquetar_region_toda_lesion():
    assert etiquetar_region(np.ones((10, 10), dtype=int)) == 1


def test_etiquetar_region_toda_sana():
    assert etiquetar_region(np.zeros((10, 10), dtype=int)) == 0


def test_etiquetar_region_region_vacia():
    assert etiquetar_region(np.array([], dtype=int)) == 0


def test_etiquetar_region_umbral_estricto():
    mascara_region = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=int)  # 40% lesión
    assert etiquetar_region(mascara_region, umbral=0.5) == 0  # no llega a 50%
    assert etiquetar_region(mascara_region, umbral=0.3) == 1  # sí llega a 30%


def test_etiquetar_region_umbral_laxo():
    mascara_region = np.array([1, 0, 0, 0, 0], dtype=int)  # 20% lesión
    assert etiquetar_region(mascara_region, umbral=0.1) == 1
    assert etiquetar_region(mascara_region, umbral=0.5) == 0


def test_etiquetar_region_retorna_entero():
    etiqueta = etiquetar_region(np.array([1, 0, 1, 1]))
    assert isinstance(etiqueta, (int, np.integer))


# ---------------------------------------------------------------------------
# etiquetar_superpixeles
# ---------------------------------------------------------------------------

def test_etiquetar_superpixeles_forma_salida():
    mascara = np.zeros((10, 10), dtype=int)
    mascara[:, :5] = 1  # mitad izquierda es lesión

    region_izq = np.zeros((10, 10), dtype=bool)
    region_izq[:, :5] = True
    region_der = np.zeros((10, 10), dtype=bool)
    region_der[:, 5:] = True

    y = etiquetar_superpixeles([region_izq, region_der], mascara, umbral=0.5)

    assert y.shape == (2,)
    assert list(y) == [1, 0]


def test_etiquetar_superpixeles_respeta_umbral():
    mascara = np.zeros((10, 10), dtype=int)
    mascara[:, :3] = 1  # solo 30% de la imagen es lesión

    region = np.ones((10, 10), dtype=bool)  # una sola región = toda la imagen

    y_laxo = etiquetar_superpixeles([region], mascara, umbral=0.2)
    y_estricto = etiquetar_superpixeles([region], mascara, umbral=0.5)

    assert y_laxo[0] == 1    # 30% >= 20%
    assert y_estricto[0] == 0  # 30% < 50%


# ---------------------------------------------------------------------------
# entrenar_modelo
# ---------------------------------------------------------------------------

@pytest.fixture
def datos_sinteticos():
    """Dos clústeres linealmente separables para probar clasificación."""
    rng = np.random.default_rng(42)
    clase_0 = rng.normal(loc=-2.0, scale=0.5, size=(30, 5))
    clase_1 = rng.normal(loc=2.0, scale=0.5, size=(30, 5))

    X = np.vstack([clase_0, clase_1])
    y = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])
    return X, y


@pytest.mark.parametrize("tipo_modelo", ["svm", "knn", "random_forest"])
def test_entrenar_modelo_tipos_soportados(datos_sinteticos, tipo_modelo):
    X, y = datos_sinteticos
    modelo = entrenar_modelo(X, y, tipo_modelo=tipo_modelo)

    assert modelo is not None
    assert hasattr(modelo, "predict")  # debe quedar entrenado y ser usable


def test_entrenar_modelo_tipo_invalido(datos_sinteticos):
    X, y = datos_sinteticos
    with pytest.raises(ValueError):
        entrenar_modelo(X, y, tipo_modelo="modelo_inexistente")


def test_entrenar_modelo_aprende_separacion_simple(datos_sinteticos):
    X, y = datos_sinteticos
    modelo = entrenar_modelo(X, y, tipo_modelo="random_forest")

    # Sobre los mismos datos de entrenamiento, un modelo razonable
    # debe acertar la gran mayoría en un problema linealmente separable.
    y_pred = modelo.predict(X)
    exactitud = np.mean(y_pred == y)
    assert exactitud > 0.9


# ---------------------------------------------------------------------------
# predecir
# ---------------------------------------------------------------------------

def test_predecir_dimensiones(datos_sinteticos):
    X, y = datos_sinteticos
    modelo = entrenar_modelo(X, y, tipo_modelo="knn")

    y_pred = predecir(modelo, X)
    assert y_pred.shape == y.shape


def test_predecir_valores_binarios(datos_sinteticos):
    X, y = datos_sinteticos
    modelo = entrenar_modelo(X, y, tipo_modelo="svm")

    y_pred = predecir(modelo, X)
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_predecir_consistente_entre_llamadas(datos_sinteticos):
    X, y = datos_sinteticos
    modelo = entrenar_modelo(X, y, tipo_modelo="random_forest")

    y_pred_1 = predecir(modelo, X)
    y_pred_2 = predecir(modelo, X)
    np.testing.assert_array_equal(y_pred_1, y_pred_2)