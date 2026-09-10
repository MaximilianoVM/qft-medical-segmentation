import numpy as np
from src.slic_custom import slic as slic_custom

def presegmentar_ventanas(imagen: np.ndarray, tam_ventana: int = 16) -> list[np.ndarray]:
    """Divide la imagen en ventanas de tamaño fijo."""
    alto, ancho = imagen.shape[:2]
    regiones = []
    for y in range(0, alto - tam_ventana + 1, tam_ventana):
        for x in range(0, ancho - tam_ventana + 1, tam_ventana):
            regiones.append(imagen[y:y + tam_ventana, x:x + tam_ventana])
    return regiones

def presegmentar_superpixeles(imagen: np.ndarray, n_segmentos: int = 100, m: float = 10, num_iters: int = 10) -> list[np.ndarray]:
    """Aplica tu implementación propia de SLIC. Devuelve lista de máscaras booleanas."""
    imagen_float = imagen / 255.0 if imagen.max() > 1 else imagen
    etiquetas = slic_custom(imagen_float, k=n_segmentos, m=m, num_iters=num_iters)
    regiones = [(etiquetas == etiqueta) for etiqueta in np.unique(etiquetas)]
    return regiones