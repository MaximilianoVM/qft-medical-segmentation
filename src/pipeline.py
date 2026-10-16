from pathlib import Path
from typing import Optional
import random
import typer
import joblib
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

# Módulos del proyecto
from src.io_utils import cargar_imagen, cargar_mascara, listar_pares_imagen_mascara
from src.presegmentation import presegmentar_superpixeles
from src.quaternion_qft import procesar_qft_superpixeles
from src.feature_extraction import extraer_matriz_caracteristicas, normalizar_caracteristicas
from src.classification import entrenar_modelo, etiquetar_superpixeles, predecir
from src.evaluation import reconstruir_mascara, calcular_metricas

cli = typer.Typer(help="CLI para el Pipeline de Segmentación Médica QFT")
console = Console()

@cli.command()
def version():
    """Muestra la versión del pipeline."""
    console.print("[bold green]QFT Medical Segmentation CLI v1.0.0[/bold green]")

@cli.command()
def entrenar(
    dir_imagenes: str = typer.Option("data/raw/train", "--dir-img", help="Directorio de imágenes de entrenamiento"),
    dir_mascaras: str = typer.Option("data/masks/train", "--dir-mask", help="Directorio de máscaras Ground-Truth"),
    salida: str = typer.Option("modelos/modelo_prueba.joblib", "--salida", "-o", help="Ruta donde guardar el .joblib"),
    limite: int = typer.Option(10, "--limite", "-n", help="Número N de imágenes para entrenar la muestra"),
    n_segmentos: int = typer.Option(100, "-k", help="Número de superpíxeles K por imagen"),
    umbral: float = typer.Option(0.5, "--umbral", help="Umbral de etiquetado de lesión"),
    tipo_modelo: str = typer.Option("random_forest", "--modelo-tipo", help="Tipo de clasificador: random_forest, svm, knn")
):
    """
    Entrena un modelo con una muestra de N imágenes y lo guarda en formato .joblib.
    """
    pares = listar_pares_imagen_mascara(dir_imagenes, dir_mascaras)
    if not pares:
        console.print("[bold red][ERROR] No se encontraron pares de imagen y máscara en los directorios indicados.[/bold red]")
        raise typer.Exit(1)

    muestra = random.sample(pares, min(limite, len(pares)))
    console.print(f"[bold blue][*][/bold blue] Entrenando modelo con muestra de [cyan]{len(muestra)}[/cyan] imágenes...")

    X_list, y_list = [], []

    for idx, (img_path, mask_path) in enumerate(muestra, 1):
        console.print(f"  [dim]Processing [{idx}/{len(muestra)}]: {Path(img_path).name}[/dim]")
        img_rgb = cargar_imagen(img_path)
        gt = cargar_mascara(mask_path)

        sp = presegmentar_superpixeles(img_rgb, n_segmentos=n_segmentos)
        esp = procesar_qft_superpixeles(img_rgb, sp)
        X_raw = extraer_matriz_caracteristicas(esp)
        y_sp = etiquetar_superpixeles(sp, gt, umbral=umbral)

        X_list.append(X_raw)
        y_list.append(y_sp)

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    console.print(f"[*] Normalizando {X_all.shape[0]} vectores de características...")
    X_norm = normalizar_caracteristicas(X_all, metodo="zscore")

    console.print(f"[*] Ajustando clasificador [bold magenta]{tipo_modelo}[/bold magenta]...")
    clf = entrenar_modelo(X_norm, y_all, tipo_modelo=tipo_modelo)

    # Crear carpeta de modelos si no existe
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, salida)
    console.print(f"[bold green][✓] Modelo guardado exitosamente en: {salida}[/bold green]")

@cli.command()
def segmentar(
    imagen: str = typer.Option(..., "--imagen", "-i", help="Ruta de la imagen RGB (.jpg)"),
    mascara: Optional[str] = typer.Option(None, "--mascara", "-m", help="Ruta del Ground-Truth (.png)"),
    modelo: str = typer.Option("modelos/modelo_prueba.joblib", "--modelo", help="Ruta del modelo .joblib"),
    salida: str = typer.Option("mascara_predicha.png", "--salida", "-o", help="Ruta de la imagen resultante"),
    n_segmentos: int = typer.Option(100, "-k", help="Número de superpíxeles K")
):
    """
    Segmenta una sola imagen utilizando el modelo .joblib indicado.
    """
    if not Path(modelo).exists():
        console.print(f"[bold red][ERROR] No existe el modelo en: {modelo}. Entrena uno primero con 'entrenar'.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[*] Procesando imagen: [cyan]{imagen}[/cyan]")
    img_rgb = cargar_imagen(imagen)
    sp = presegmentar_superpixeles(img_rgb, n_segmentos=n_segmentos)
    esp = procesar_qft_superpixeles(img_rgb, sp)
    X_norm = normalizar_caracteristicas(extraer_matriz_caracteristicas(esp))

    clf = joblib.load(modelo)
    y_pred = predecir(clf, X_norm)

    h, w, _ = img_rgb.shape
    mask_pred = reconstruir_mascara(sp, y_pred, forma=(h, w))
    plt.imsave(salida, mask_pred, cmap="gray")
    console.print(f"[bold green][✓] Máscara reconstruida en: {salida}[/bold green]")

    if mascara and Path(mascara).exists():
        gt = cargar_mascara(mascara)
        m = calcular_metricas(mask_pred, gt)
        console.print(f"[bold cyan]IoU:[/bold cyan] {m['iou']:.4f} | [bold cyan]Dice:[/bold cyan] {m['dice']:.4f}")

@cli.command()
def evaluar_lote(
    dir_imagenes: str = typer.Option("data/raw/test", "--dir-img", help="Directorio de imágenes de test"),
    dir_mascaras: str = typer.Option("data/masks/test", "--dir-mask", help="Directorio de máscaras de test"),
    modelo: str = typer.Option("modelos/modelo_prueba.joblib", "--modelo", help="Ruta del modelo .joblib"),
    limite: int = typer.Option(5, "--limite", "-n", help="Número N de imágenes a evaluar"),
    n_segmentos: int = typer.Option(100, "-k", help="Número de superpíxeles K")
):
    """
    Evalúa el desempeño en una muestra aleatoria de N imágenes de prueba.
    """
    if not Path(modelo).exists():
        console.print(f"[bold red][ERROR] No existe el modelo en: {modelo}[/bold red]")
        raise typer.Exit(1)

    pares = listar_pares_imagen_mascara(dir_imagenes, dir_mascaras)
    muestra = random.sample(pares, min(limite, len(pares)))
    clf = joblib.load(modelo)

    ious, dices = [], []
    for img_path, mask_path in muestra:
        img_rgb = cargar_imagen(img_path)
        gt = cargar_mascara(mask_path)
        sp = presegmentar_superpixeles(img_rgb, n_segmentos=n_segmentos)
        esp = procesar_qft_superpixeles(img_rgb, sp)
        X_norm = normalizar_caracteristicas(extraer_matriz_caracteristicas(esp))

        y_pred = predecir(clf, X_norm)
        mask_pred = reconstruir_mascara(sp, y_pred, forma=gt.shape)
        m = calcular_metricas(mask_pred, gt)
        ious.append(m["iou"])
        dices.append(m["dice"])

    table = Table(title=f"Evaluación en Muestra (N={len(muestra)})")
    table.add_column("Métrica", style="cyan")
    table.add_column("Resultado", style="magenta")
    table.add_row("IoU Promedio", f"{np.mean(ious):.4f} ± {np.std(ious):.4f}")
    table.add_row("Dice Promedio", f"{np.mean(dices):.4f} ± {np.std(dices):.4f}")
    console.print(table)

if __name__ == "__main__":
    cli()
