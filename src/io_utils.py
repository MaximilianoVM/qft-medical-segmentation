import numpy as np

def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga una imagen RGB y la devuelve como array (H, W, 3)."""
    pass

def cargar_mascara(ruta: str) -> np.ndarray:
    """Carga la máscara de referencia como array (H, W) con etiquetas de clase."""
    pass

def listar_pares_imagen_mascara(dir_imagenes: str, dir_mascaras: str) -> list[tuple[str, str]]:
    """Devuelve pares (ruta_imagen, ruta_mascara) que coinciden por nombre de archivo."""
    pass