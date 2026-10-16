from pathlib import Path
import sys

# Agregar la raíz del proyecto a la ruta de búsqueda de Python
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
  sys.path.insert(0, str(RAIZ_PROYECTO))

import joblib
import matplotlib.pyplot as plt
import random
from src.classification import predecir
from src.evaluation import calcular_metricas, reconstruir_mascara
from src.feature_extraction import (
    extraer_matriz_caracteristicas,
    normalizar_caracteristicas,
)
from src.io_utils import cargar_imagen, cargar_mascara, listar_pares_imagen_mascara
from src.presegmentation import presegmentar_superpixeles
from src.quaternion_qft import procesar_qft_superpixeles
import streamlit as st

st.set_page_config(page_title="QFT medical segmentation", layout="wide")

st.title("🔬 Segmentación de Lesiones Dermatológicas mediante QFT 2D")
st.sidebar.header("Configuración de Control")

# Cargar modelo entrenado
modelo_path = st.sidebar.file_uploader("Cargar archivo de modelo (.joblib)", type=["joblib"])
k_sp = st.sidebar.slider("Número de Superpíxeles (K)", min_value=30, max_value=200, value=100)

if st.button("🎲 Cargar Imagen Aleatoria del Test Set"):
    pares = listar_pares_imagen_mascara("data/raw/test", "data/masks/test")
    img_path, mask_path = random.choice(pares)
    st.session_state["img_path"] = img_path
    st.session_state["mask_path"] = mask_path

if "img_path" in st.session_state:
    col1, col2, col3 = st.columns(3)
    
    img_rgb = cargar_imagen(st.session_state["img_path"])
    gt = cargar_mascara(st.session_state["mask_path"])
    
    with col1:
        st.subheader("Imagen Dermatológica (RGB)")
        st.image(img_rgb, use_container_width=True)
        
    with col2:
        st.subheader("Máscara Ground-Truth")
        st.image(gt, use_container_width=True)

    if modelo_path is not None:
        clf = joblib.load(modelo_path)
        
        # Procesamiento QFT
        with st.spinner("Procesando QFT y prediciendo lesión..."):
            sp = presegmentar_superpixeles(img_rgb, n_segmentos=k_sp)
            esp = procesar_qft_superpixeles(img_rgb, sp)
            X_norm = normalizar_caracteristicas(extraer_matriz_caracteristicas(esp))
            y_pred = predecir(clf, X_norm)
            mask_pred = reconstruir_mascara(sp, y_pred, forma=gt.shape)
            m = calcular_metricas(mask_pred, gt)
            
        with col3:
            st.subheader("Predicción QFT Reconstruida")
            st.image(mask_pred * 255, use_container_width=True)
            
        st.success(f"**Resultados:** IoU = **{m['iou']:.4f}** | Dice = **{m['dice']:.4f}**")
