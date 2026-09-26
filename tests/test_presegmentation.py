import numpy as np
from src.presegmentation import presegmentar_superpixeles

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