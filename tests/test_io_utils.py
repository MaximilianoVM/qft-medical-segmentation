import numpy as np
from PIL import Image
from src.io_utils import cargar_imagen, cargar_mascara, listar_pares_imagen_mascara

def test_cargar_imagen(tmp_path):
    ruta = tmp_path / "test.jpg"
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(ruta)

    resultado = cargar_imagen(str(ruta))
    assert resultado.shape == (10, 10, 3)
    # JPEG es compresión con pérdida: toleramos una pequeña diferencia
    assert np.allclose(resultado[0, 0], [255, 0, 0], atol=5)

def test_cargar_mascara(tmp_path):
    ruta = tmp_path / "mask.png"
    mask = Image.new("L", (10, 10), color=1)
    mask.save(ruta)

    resultado = cargar_mascara(str(ruta))
    assert resultado.shape == (10, 10)

def test_listar_pares(tmp_path):
    dir_img = tmp_path / "imgs"
    dir_mask = tmp_path / "masks"
    dir_img.mkdir()
    dir_mask.mkdir()

    # Simula la convención real del dataset ISIC
    Image.new("RGB", (5, 5)).save(dir_img / "ISIC_0000001.jpg")
    Image.new("L", (5, 5)).save(dir_mask / "ISIC_0000001_Segmentation.png")

    pares = listar_pares_imagen_mascara(str(dir_img), str(dir_mask))
    assert len(pares) == 1
    assert "ISIC_0000001.jpg" in pares[0][0]
    assert "ISIC_0000001_Segmentation.png" in pares[0][1]