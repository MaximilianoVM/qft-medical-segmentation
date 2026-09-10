import numpy as np

def presegmentar_ventanas(imagen: np.ndarray, tam_ventana: int = 16) -> list[np.ndarray]:
    """Divide la imagen en ventanas de tamaño fijo. Devuelve lista de regiones (sub-arrays)."""
    pass

def presegmentar_superpixeles(imagen: np.ndarray, n_segmentos: int = 100) -> list[np.ndarray]:
    """Aplica superpíxeles (ej. SLIC) y devuelve lista de regiones."""
    pass