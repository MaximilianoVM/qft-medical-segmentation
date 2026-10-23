# src/generate_figures.py
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm

from src.io_utils import cargar_imagen, cargar_mascara, listar_pares_imagen_mascara
from src.presegmentation import presegmentar_superpixeles
from src.quaternion_qft import procesar_qft_superpixeles
from src.feature_extraction import extraer_matriz_caracteristicas, normalizar_caracteristicas
from src.classification import etiquetar_superpixeles, predecir
from src.evaluation import reconstruir_mascara, calcular_metricas

_CLF = None


def _get_clf(ruta_modelo):
    """Carga el modelo una vez por proceso (no una vez por tarea)."""
    global _CLF
    if _CLF is None:
        _CLF = joblib.load(ruta_modelo)
        if hasattr(_CLF, "n_jobs"):
            _CLF.n_jobs = 1  # evita sobresuscripción de CPU
    return _CLF


def _pipeline(par, clf, n_segmentos):
    """Pipeline completo para una imagen. Devuelve también los arreglos pesados."""
    r_img, r_mask = par
    img, mask = cargar_imagen(r_img), cargar_mascara(r_mask)
    sp = presegmentar_superpixeles(img, n_segmentos=n_segmentos)
    qft = procesar_qft_superpixeles(img, sp)
    X_norm = normalizar_caracteristicas(extraer_matriz_caracteristicas(qft), metodo="zscore")
    y_gt = etiquetar_superpixeles(sp, mask, umbral=0.9)
    y_p = predecir(clf, X_norm)
    m_pred = reconstruir_mascara(sp, y_p, forma=img.shape[:2])
    return img, mask, y_gt, y_p, m_pred


def _memoria_disponible():
    """Bytes de RAM disponibles (Linux /proc/meminfo, también en Docker); None si no se puede leer."""
    try:
        with open("/proc/meminfo") as f:
            for linea in f:
                if linea.startswith("MemAvailable:"):
                    return int(linea.split()[1]) * 1024
    except OSError:
        pass
    return None


def _elegir_n_jobs(n_jobs_cli, ruta_modelo):
    """Prioridad: --jobs, luego N_JOBS, luego estimación según RAM disponible."""
    if n_jobs_cli is not None:
        return n_jobs_cli, "--jobs"
    if os.environ.get("N_JOBS"):
        return int(os.environ["N_JOBS"]), "variable N_JOBS"
    mem = _memoria_disponible()
    if mem is None:
        return 1, "RAM disponible desconocida"
    # Medido: un worker llega a ~700 MB (imports + pipeline); se suma el modelo
    # deserializado (~3x su tamaño en disco) y solo se usa el 70% de la RAM libre.
    por_worker = 800 * 1024**2 + 3 * os.path.getsize(ruta_modelo)
    n = max(1, min(os.cpu_count() or 1, int(0.7 * mem) // por_worker))
    return n, f"{mem / 1024**3:.1f} GB disponibles, ~{por_worker / 1024**2:.0f} MB por worker"


def _procesar_par_imagen(par, ruta_modelo, n_segmentos):
    """Worker: devuelve solo datos ligeros (sin imágenes)."""
    try:
        clf = _get_clf(ruta_modelo)
        img, mask, y_gt, y_p, m_pred = _pipeline(par, clf, n_segmentos)
        met = calcular_metricas(m_pred, mask)
        return {
            "y_gt": np.asarray(y_gt, dtype=np.uint8),
            "y_p": np.asarray(y_p, dtype=np.uint8),
            # máscara predicha comprimida a bits: evita recalcular el pipeline para mejor/peor caso
            "m_pred": np.packbits(np.asarray(m_pred, dtype=bool)),
            "forma": m_pred.shape,
            "metrics": {**met, "r_img": par[0], "r_mask": par[1]},
        }
    except Exception as e:
        print(f"Error procesando {par[0]}: {e}")
        return None


def generar_matriz_confusion_modelo(resultados_eval, dir_figures: Path):
    print("📊 Generando matriz de confusión del modelo...")
    y_reales, y_predichas = [], []
    for res in resultados_eval:  # todas las imágenes evaluadas
        y_reales.extend(res["y_gt"])
        y_predichas.extend(res["y_p"])

    cm = confusion_matrix(y_reales, y_predichas)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["piel sana", "lesión"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Matriz de Confusión (Random Forest QFT)", fontweight="bold")
    plt.tight_layout()
    out_path = dir_figures / "confusion_matrices.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardada en: {out_path}")


def evaluar_test_y_generar_histogramas(resultados_eval, dir_figures: Path):
    print("📊 Generando histogramas y comparativa de casos...")
    df = pd.DataFrame([r["metrics"] for r in resultados_eval])

    # 1. Histogramas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, titulo in [(axes[0], "iou", "IoU"), (axes[1], "dice", "Dice")]:
        ax.hist(df[col], bins=10, edgecolor="black")
        ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"media ({df[col].mean():.2f})")
        ax.set_title(f"Distribución de {titulo}", fontweight="bold")
        ax.set_xlabel(titulo)
        ax.legend()
    plt.tight_layout()
    out_hist = dir_figures / "metrics_distribution.png"
    plt.savefig(out_hist, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado en: {out_hist}")

    # 2. Mejor / peor caso: se usa la máscara predicha guardada (sin volver a cargar el modelo)
    def _caso(idx):
        row = df.loc[idx]
        res = resultados_eval[idx]
        alto, ancho = res["forma"]
        pred = np.unpackbits(res["m_pred"], count=alto * ancho).reshape(alto, ancho)
        return row, cargar_imagen(row["r_img"]), cargar_mascara(row["r_mask"]), pred

    casos = [
        ("Mejor Caso", *_caso(df["iou"].idxmax())),
        ("Peor Caso", *_caso(df["iou"].idxmin())),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (nombre, row, img, mask, pred) in enumerate(casos):
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{nombre}: {Path(row['r_img']).name}")
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title(f"Predicción (IoU={row['iou']:.2f}, Dice={row['dice']:.2f})")
    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    out_cases = dir_figures / "best_worst_cases.png"
    plt.savefig(out_cases, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardado en: {out_cases}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera todas las figuras necesarias para el reporte JOSS")
    parser.add_argument("-m", "--modelo", type=str, required=True, help="Ruta al archivo .joblib del modelo")
    parser.add_argument("-k", "--superpixeles", type=int, default=85, help="Número de superpíxeles")
    parser.add_argument("-n", "--eval-count", type=int, default=30, help="Número de imágenes a evaluar")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Procesos en paralelo (por defecto: N_JOBS o estimado según la RAM disponible)")
    args = parser.parse_args()

    dir_figures = Path("paper/figures")
    dir_figures.mkdir(parents=True, exist_ok=True)

    pares_test = listar_pares_imagen_mascara("data/raw/test", "data/masks/test")[:args.eval_count]

    n_jobs, motivo = _elegir_n_jobs(args.jobs, args.modelo)
    print(f"📊 Procesando {len(pares_test)} imágenes con {n_jobs} workers ({motivo})...")
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_procesar_par_imagen)(p, args.modelo, args.superpixeles) for p in pares_test
    )
    res_raw = list(tqdm(gen, total=len(pares_test), desc="Evaluando"))
    resultados = [r for r in res_raw if r is not None]
    # Libera los workers (y sus copias del modelo) antes de dibujar las figuras
    get_reusable_executor().shutdown(wait=True)

    if resultados:
        generar_matriz_confusion_modelo(resultados, dir_figures)
        evaluar_test_y_generar_histogramas(resultados, dir_figures)
        print("✨ ¡Todas las figuras del reporte fueron generadas con éxito!")
    else:
        print("❌ No se pudieron procesar las imágenes.")