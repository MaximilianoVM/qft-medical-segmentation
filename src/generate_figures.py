import sys
from pathlib import Path

# Añade la raíz del proyecto al sys.path para permitir imports del tipo 'from src.module import ...'
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import argparse
import joblib

from src.io_utils import cargar_imagen, cargar_mascara
from src.presegmentation import presegmentar_superpixeles
from src.quaternion_qft import procesar_qft_superpixeles
from src.feature_extraction import extraer_matriz_caracteristicas, normalizar_caracteristicas
from src.classification import predecir
from src.evaluation import reconstruir_mascara, plot_segmentation_results


def generar_figura_paper(
    ruta_modelo: str,
    ruta_img: str = "data/raw/test/ISIC_0000003.jpg",
    ruta_mask: str = "data/masks/test/ISIC_0000003_Segmentation.png",
    ruta_salida: str = "paper/figures/segmentation_example.png",
    n_segmentos: int = 85
):
    """
    Genera la figura de segmentación comparativa para el reporte a partir de un modelo .joblib dado.
    """
    path_mod = Path(ruta_modelo)
    if not path_mod.is_file():
        print(f"❌ Error: El archivo de modelo '{ruta_modelo}' no existe.", file=sys.stderr)
        print("   Entrena primero un modelo o especifica una ruta válida con --modelo", file=sys.stderr)
        sys.exit(1)

    path_out = Path(ruta_salida)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Cargando modelo desde: {ruta_modelo}")
    clf = joblib.load(path_mod)

    img_rgb = cargar_imagen(ruta_img)
    gt_mask = cargar_mascara(ruta_mask)

    print("🔄 Ejecutando pipeline QFT + SLIC...")
    regiones_sp = presegmentar_superpixeles(img_rgb, n_segmentos=n_segmentos)
    espectros_qft = procesar_qft_superpixeles(img_rgb, regiones_sp)
    X_raw = extraer_matriz_caracteristicas(espectros_qft)
    X_norm = normalizar_caracteristicas(X_raw, metodo="zscore")

    y_pred = predecir(clf, X_norm)
    h, w, _ = img_rgb.shape
    pred_mask = reconstruir_mascara(regiones_sp, y_pred, forma=(h, w))

    plot_segmentation_results(img_rgb, gt_mask, pred_mask, save_path=str(path_out))
    print(f"✅ Figura generada exitosamente en: {path_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera figuras para el reporte a partir de un modelo .joblib")
    parser.add_argument("-m", "--modelo", type=str, required=True, help="Ruta al archivo .joblib del modelo entrenado")
    parser.add_argument("-i", "--imagen", type=str, default="data/raw/test/ISIC_0000003.jpg", help="Ruta a la imagen de prueba")
    parser.add_argument("-g", "--mascara", type=str, default="data/masks/test/ISIC_0000003_Segmentation.png", help="Ruta a la máscara Ground Truth")
    parser.add_argument("-o", "--output", type=str, default="paper/figures/segmentation_example.png", help="Ruta de salida del PNG")
    parser.add_argument("-k", "--superpixeles", type=int, default=85, help="Número de superpíxeles SLIC")

    args = parser.parse_args()

    generar_figura_paper(
        ruta_modelo=args.modelo,
        ruta_img=args.imagen,
        ruta_mask=args.mascara,
        ruta_salida=args.output,
        n_segmentos=args.superpixeles
    )
