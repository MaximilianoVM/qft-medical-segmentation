import numpy as np
from src.presegmentation import presegmentar_ventanas, presegmentar_superpixeles

def test_presegmentar_ventanas():
    imagen = np.zeros((20, 20, 3), dtype=np.uint8)
    regiones = presegmentar_ventanas(imagen, tam_ventana=10)
    # Una imagen de 20x20 con ventanas de 10x10 da 2x2 = 4 regiones
    assert len(regiones) == 4
    for region in regiones:
        assert region.shape == (10, 10, 3)

def test_presegmentar_ventanas_descarta_sobrante():
    imagen = np.zeros((25, 25, 3), dtype=np.uint8)
    regiones = presegmentar_ventanas(imagen, tam_ventana=10)
    # 25 no es múltiplo de 10: solo caben 2 ventanas completas por eje, se descarta el sobrante
    assert len(regiones) == 4

def test_presegmentar_superpixeles():
    imagen = np.random.randint(0, 255, size=(20, 20, 3), dtype=np.uint8)
    regiones = presegmentar_superpixeles(imagen, n_segmentos=4, num_iters=2)

    assert len(regiones) > 0
    for mascara in regiones:
        assert mascara.shape == (20, 20)
        assert mascara.dtype == bool

    # Cada pixel debe pertenecer a exactamente una región (partición completa, sin huecos ni traslapes)
    suma = np.zeros((20, 20), dtype=int)
    for mascara in regiones:
        suma += mascara.astype(int)
    assert (suma == 1).all()