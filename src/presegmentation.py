import numpy as np
from src.slic_custom import slic as slic_custom
from skimage.segmentation import slic as slic_skimage

#def presegmentar_ventanas(imagen: np.ndarray, tam_ventana: int = 16) -> list[np.ndarray]:
#    pass

def presegmentar_superpixeles(imagen: np.ndarray, n_segmentos: int = 100, m: float = 10, num_iters: int = 10,
                              metodo: str = "propio") -> list[np.ndarray]:
    '''
    k = n_segmentos -> número de superpíxeles deseados
    m ∈ [1,40]-> importancia relativa entre similitud de color (↓) y proximidad espacial (↑)
    metodo -> "propio" (nuestra SLIC, src/slic_custom.py) o "skimage" (skimage.segmentation.slic,
              más rápida; da superpíxeles distintos, así que requiere su propio modelo entrenado)

    '''
    
    # aplicamos SLIC. Devuelve lista de máscaras booleanas
    imagen_float = imagen / 255.0 if imagen.max() > 1 else imagen
    if metodo == "propio":
        etiquetas = slic_custom(imagen_float, k=n_segmentos, m=m, num_iters=num_iters)
    elif metodo == "skimage":
        etiquetas = slic_skimage(imagen_float, n_segments=n_segmentos, compactness=m,
                                 max_num_iter=num_iters, start_label=0)
    else:
        raise ValueError(f"Método no soportado: {metodo}. Usar 'propio' o 'skimage'.")
    regiones = [(etiquetas == etiqueta) for etiqueta in np.unique(etiquetas)]
    return regiones

def region_a_recorte_rgb(imagen: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    # convierte una mascara booleana (tamaño completo) en un recorte RGB
    filas, columnas = np.where(mascara)
    y0, y1 = filas.min(), filas.max() + 1
    x0, x1 = columnas.min(), columnas.max() + 1

    recorte = imagen[y0:y1, x0:x1].copy()
    mascara_recorte = mascara[y0:y1, x0:x1]
    recorte[~mascara_recorte] = 0  # apaga los píxeles que no son de este superpíxel
    return recorte