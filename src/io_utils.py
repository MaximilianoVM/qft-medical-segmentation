import numpy as np
from pathlib import Path
from PIL import Image

'''
io: i-ns and o-uts

utils: utilities

1er paso, aqui nos encargamos unicamente de leer los archivos y hacerlos 
manipulables para python en los pasos posteriores
'''

def cargar_imagen(ruta: str) -> np.ndarray:
    # Carga una imagen RGB y la devuelve como array (H, W, 3).
    imagen = Image.open(ruta).convert("RGB")
    return np.array(imagen)

def cargar_mascara(ruta: str) -> np.ndarray:
    # Carga la máscara de referencia como array (H, W) con etiquetas de clase.
    mascara = Image.open(ruta).convert("L")  # escala de grises = etiquetas
    return np.array(mascara)

def listar_pares_imagen_mascara(dir_imagenes: str, dir_mascaras: str) -> list[tuple[str, str]]:
    # Devuelve pares (ruta_imagen, ruta_mascara) que coinciden por nombre de archivo.
    dir_img = Path(dir_imagenes)
    dir_mask = Path(dir_mascaras)
    pares = []
    for archivo_img in sorted(dir_img.glob("*.jpg")):
        id_imagen = archivo_img.stem  # ej. "ISIC_0000003"
        candidato_mascara = dir_mask / f"{id_imagen}_Segmentation.png"
        if candidato_mascara.exists():
            pares.append((str(archivo_img), str(candidato_mascara)))
    return pares