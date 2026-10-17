from pathlib import Path
import sys

# 1. Asegurar la raíz del proyecto en sys.path
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from skimage.segmentation import mark_boundaries

# Importación de módulos del proyecto
from src.io_utils import cargar_imagen, cargar_mascara, listar_pares_imagen_mascara
from src.presegmentation import presegmentar_superpixeles
from src.quaternion_qft import procesar_qft_superpixeles
from src.feature_extraction import extraer_matriz_caracteristicas, normalizar_caracteristicas
from src.classification import predecir
from src.evaluation import reconstruir_mascara, calcular_metricas

# Configuración de página
st.set_page_config(
    page_title="QFT Medical Skin Segmentation",
    page_icon="🔬",
    layout="wide"
)

# Función con caché para cargar el modelo de 189 MB una sola vez en la memoria RAM
@st.cache_resource
def cargar_modelo_cached(ruta_modelo: str):
    return joblib.load(ruta_modelo)

st.title("🔬 Sistema de Segmentación de Lesiones Cutáneas (QFT + SLIC)")
st.caption("Prototipo de Inferencia Clínica para Dermatología | Basado en ISIC Archive")

# --- BARRA LATERAL (CONFIGURACIÓN) ---
st.sidebar.header("⚙️ Panel de Control")

# 1. Carga del Modelo Preentrenado (.joblib)
st.sidebar.subheader("1. Modelo Clasificador")
archivo_modelo = st.sidebar.file_uploader("Cargar modelo (.joblib)", type=["joblib"])

# Detectar modelo por defecto o modelo subido por el usuario
modelo_defecto = Path("modelos/modelo_prueba.joblib")
if archivo_modelo is None and modelo_defecto.exists():
    path_modelo_usar = str(modelo_defecto)
    st.sidebar.info("Uso de modelo local: `modelo_prueba.joblib`")
elif archivo_modelo is not None:
    temp_model_path = Path("modelos/temp_uploaded_model.joblib")
    temp_model_path.parent.mkdir(exist_ok=True)
    with open(temp_model_path, "wb") as f:
        f.write(archivo_modelo.getbuffer())
    path_modelo_usar = str(temp_model_path)
    st.sidebar.success(f"Modelo cargado: {archivo_modelo.name} ({archivo_modelo.size / 1e6:.1f} MB)")
else:
    path_modelo_usar = None

# 2. Configuración de Superpíxeles (K)
st.sidebar.subheader("2. Parámetros de Presegmentación")
k_superpixeles = st.sidebar.slider(
    "Número de Superpíxeles (K)",
    min_value=30,
    max_value=200,
    value=100,
    step=10,
    help="100 es el valor estándar utilizado en la Fase 4."
)

# 3. Selección de la Fuente de Imagen
st.sidebar.subheader("3. Fuente de la Imagen")
modo_entrada = st.sidebar.radio(
    "Modo de uso:",
    ["Probar dataset de evaluación (Test Set)", "Subir imagen del médico (.jpg)"]
)

img_rgb = None
mascara_gt = None
nombre_imagen = ""

if modo_entrada == "Probar dataset de evaluación (Test Set)":
    pares_test = listar_pares_imagen_mascara("data/raw/test", "data/masks/test")
    if pares_test:
        if st.sidebar.button("🎲 Seleccionar Imagen Aleatoria"):
            par_sel = random.choice(pares_test)
            st.session_state["par_sel"] = par_sel
            st.session_state.pop("mask_pred", None) # Limpiar predicción anterior
            
        if "par_sel" in st.session_state:
            ruta_i, ruta_m = st.session_state["par_sel"]
            img_rgb = cargar_imagen(ruta_i)
            mascara_gt = cargar_mascara(ruta_m)
            nombre_imagen = Path(ruta_i).name
            st.sidebar.success(f"Imagen seleccionada: `{nombre_imagen}`")
    else:
        st.sidebar.error("No se encontraron pares en `data/raw/test/`.")

else:
    uploaded_file = st.sidebar.file_uploader("Selecciona una foto dermatológica RGB", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        if "uploaded_name" not in st.session_state or st.session_state["uploaded_name"] != uploaded_file.name:
            st.session_state["uploaded_name"] = uploaded_file.name
            st.session_state.pop("mask_pred", None)
            
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        nombre_imagen = uploaded_file.name

# --- PANTALLA PRINCIPAL ---
if img_rgb is not None:
    # 1. MOSTRAR VISTA PREVIA INMEDIATA DE LA IMAGEN CARGADA
    st.subheader(f"📷 Imagen Seleccionada: {nombre_imagen}")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.image(img_rgb, caption="Imagen Dermatológica RGB", use_container_width=True)
    with col_p2:
        if mascara_gt is not None:
            st.image(mascara_gt * 255 if mascara_gt.max() <= 1 else mascara_gt, caption="Máscara Ground-Truth (Diagnóstico Real)", use_container_width=True)
        else:
            st.info("ℹ️ Imagen externa del médico (Sin Ground-Truth de referencia).")

    st.divider()

    # 2. VERIFICACIÓN DEL MODELO Y BOTÓN DE SEGMENTACIÓN
    if path_modelo_usar is None:
        st.warning("⚠️ Debes cargar un archivo `.joblib` en la barra lateral para ejecutar la segmentación.")
    else:
        if st.button("🚀 Ejecutar Segmentación QFT", type="primary", use_container_width=True):
            with st.spinner(f"Inferencia con modelo ({Path(path_modelo_usar).name}) | Calculando SLIC y QFT..."):
                clf = cargar_modelo_cached(path_modelo_usar)
                
                # SLIC -&gt; QFT -&gt; Extracción -&gt; Z-Score -&gt; Predicción
                regiones_sp = presegmentar_superpixeles(img_rgb, n_segmentos=k_superpixeles)
                espectros_qft = procesar_qft_superpixeles(img_rgb, regiones_sp)
                X_raw = extraer_matriz_caracteristicas(espectros_qft)
                X_norm = normalizar_caracteristicas(X_raw, metodo="zscore")
                y_pred = predecir(clf, X_norm)
                
                h, w, _ = img_rgb.shape
                mask_pred = reconstruir_mascara(regiones_sp, y_pred, forma=(h, w))
                
                st.session_state["mask_pred"] = mask_pred
                st.session_state["regiones_sp"] = regiones_sp

        # 3. MOSTRAR RESULTADOS PROCESADOS
        if "mask_pred" in st.session_state:
            mask_pred = st.session_state["mask_pred"]
            regiones_sp = st.session_state["regiones_sp"]
            
            st.subheader("📊 Resultados del Análisis QFT")
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.markdown("**1. Original**")
                st.image(img_rgb, use_container_width=True)
                
            with c2:
                st.markdown(f"**2. SLIC ({len(regiones_sp)} K)**")
                labels_sp = np.zeros(img_rgb.shape[:2], dtype=int)
                for idx, mask in enumerate(regiones_sp, start=1):
                    labels_sp[mask] = idx
                slic_outline = mark_boundaries(img_rgb, labels_sp, color=(1, 1, 0))
                st.image(slic_outline, use_container_width=True)
                
            with c3:
                st.markdown("**3. Predicción QFT**")
                st.image(mask_pred * 255 if mask_pred.max() <= 1 else mask_pred, use_container_width=True)
                
            with c4:
                st.markdown("**4. Contorno de Lesión**")
                img_overlay = img_rgb.copy()
                contornos, _ = cv2.findContours(mask_pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_overlay, contornos, -1, (255, 0, 0), 3)
                st.image(img_overlay, use_container_width=True)

            if mascara_gt is not None:
                st.divider()
                st.subheader("📈 Métricas de Evaluación vs Diagnóstico Real")
                m = calcular_metricas(mask_pred, mascara_gt)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("IoU (Jaccard)", f"{m['iou']:.4f}")
                m2.metric("Dice (F1-Score)", f"{m['dice']:.4f}")
                m3.metric("Precisión", f"{m['precision']:.4f}")
                m4.metric("Recall (Sensibilidad)", f"{m['recall']:.4f}")
else:
    st.info("👈 Selecciona una imagen de prueba o sube un archivo desde el panel lateral de la izquierda para comenzar.")
