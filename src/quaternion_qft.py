import numpy as np

#def imagen_a_cuaternion(region: np.ndarray) -> np.ndarray:
    # Convierte una región RGB (H, W, 3) en representación cuaterniónica q = R*i + G*j + B*k.
 #   pass

#def aplicar_qft(region_cuaternion: np.ndarray) -> np.ndarray:
    # Aplica la Transformada Cuaterniónica de Fourier sobre la región. Devuelve el espectro QFT.
 #   pass

"""
Modulo: quaternion_qft.py
Descripción:
    Implementación de los Pasos 2 y 3 del proyecto de segmentación médica:
    - Paso 2: Representación de píxeles RGB mediante cuaterniones puros:
      q(x,y) = 0 + R(x,y)*i + G(x,y)*j + B(x,y)*k
    - Paso 3: Aplicación de la Transformada Cuaterniónica de Fourier 2D (QFT)
      sobre submatrices / regiones de superpíxeles.
"""

import numpy as np


def rgb_a_cuaternion(imagen_rgb: np.ndarray) -> np.ndarray:
    """
    Paso 2: Convierte una imagen RGB de dimensiones (H, W, 3) en un arreglo
    de cuaterniones puros de dimensiones (H, W, 4).

    Representación cuaterniónica pura:
        q(x, y) = w + x*i + y*j + z*k
        donde:
            w = 0.0          (parte real nula)
            x = R(x, y)      (componente i - Rojo)
            y = G(x, y)      (componente j - Verde)
            z = B(x, y)      (componente k - Azul)

    Parámetros:
        imagen_rgb (np.ndarray): Imagen en formato RGB con valores [0, 255] o [0.0, 1.0].

    Retorna:
        np.ndarray: Arreglo de forma (H, W, 4) de tipo float64 con los cuaterniones puros.
    """
    if not isinstance(imagen_rgb, np.ndarray):
        raise TypeError("La imagen de entrada debe ser un np.ndarray.")
    if imagen_rgb.ndim != 3 or imagen_rgb.shape[2] != 3:
        raise ValueError(f"Se esperaba una imagen de forma (H, W, 3), pero se recibió {imagen_rgb.shape}.")

    # Normalizar valores a rango [0, 1] si vienen en uint8 [0, 255]
    if imagen_rgb.dtype == np.uint8:
        rgb_norm = imagen_rgb.astype(np.float64) / 255.0
    else:
        rgb_norm = imagen_rgb.astype(np.float64)

    H, W, _ = imagen_rgb.shape
    q_img = np.zeros((H, W, 4), dtype=np.float64)

    # w = 0 (parte real nula para cuaternión puro)
    q_img[:, :, 0] = 0.0
    # x = R, y = G, z = B
    q_img[:, :, 1] = rgb_norm[:, :, 0]
    q_img[:, :, 2] = rgb_norm[:, :, 1]
    q_img[:, :, 3] = rgb_norm[:, :, 2]

    return q_img

def _qft_canal_real(canal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la QFT 2D de un solo canal real (Ecuación 6 del paper).

    F(u,v) = R + iI + jJ + kK, obtenida como transformada en cascada:
    primero sobre `y` con la unidad j, luego sobre `x` con la unidad i.

    Retorna las 4 componentes reales (R, I, J, K), cada una (M, N).
    """
    # Paso sobre el eje y (unidad j): Gc - j*Gs
    Fy = np.fft.fft(canal, axis=1)
    Gc = Fy.real
    Gs = -Fy.imag

    # Paso sobre el eje x (unidad i)
    Fx_Gc = np.fft.fft(Gc, axis=0)
    Fx_Gs = np.fft.fft(Gs, axis=0)

    R = Fx_Gc.real
    I = Fx_Gc.imag
    J = -Fx_Gs.real
    K = -Fx_Gs.imag
    return R, I, J, K

def qft_2d_region(submatriz_q: np.ndarray) -> np.ndarray:
    M, N, _ = submatriz_q.shape
    if M == 0 or N == 0:
        raise ValueError("La submatriz no puede tener dimensiones nulas.")

    # Canales de color (guardados como i, j, k en el cuaternión puro de entrada)
    canal_R = submatriz_q[:, :, 1]
    canal_G = submatriz_q[:, :, 2]
    canal_B = submatriz_q[:, :, 3]

    # QFT de cada canal por separado (Ecuación 6): cada uno da (R_i, I_i, J_i, K_i)
    R1, I1, J1, K1 = _qft_canal_real(canal_R)
    R2, I2, J2, K2 = _qft_canal_real(canal_G)
    R3, I3, J3, K3 = _qft_canal_real(canal_B)

    # Combinación de los 3 canales en un solo cuaternión (Ecuación 10)
    w = -I1 - J2 + K3
    x = R1 - K2 - J3
    y = -K1 + R2 - I3
    z = J1 + I2 + R3

    factor_norm = 1.0 / np.sqrt(M * N)
    F_q = np.stack([w, x, y, z], axis=-1) * factor_norm

    return F_q


def procesar_qft_superpixeles(
    imagen_rgb: np.ndarray,
    lista_mascaras: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Función orquestadora que integra los Pasos 2 y 3 sobre todas las regiones
    o superpíxeles de una imagen.

    Proceso por cada superpíxel k:
        1. Convierte la imagen RGB completa a representación cuaterniónica pura.
        2. Extrae la Bounding Box de la región k (tamaño M_k x N_k).
        3. Aplica máscara booleana asignando 0 a píxeles fuera de la región.
        4. Calcula la QFT 2D sobre la submatriz local.

    Parámetros:
        imagen_rgb (np.ndarray): Imagen original RGB (H, W, 3).
        lista_mascaras (list[np.ndarray]): Lista de K máscaras booleanas (H, W),
                                           donde True indica pertenencia al superpíxel.

    Retorna:
        list[np.ndarray]: Lista de K matrices espectrales F_k de dimensiones (M_k, N_k, 4).
    """
    q_img = rgb_a_cuaternion(imagen_rgb)
    espectros_qft = []

    for idx, mask in enumerate(lista_mascaras):
        filas, columnas = np.where(mask)

        # Si el superpíxel está vacío, se omite o se agrega arreglo vacío
        if len(filas) == 0 or len(columnas) == 0:
            continue

        # Delimitar la Bounding Box local
        rmin, rmax = int(filas.min()), int(filas.max())
        cmin, cmax = int(columnas.min()), int(columnas.max())

        # Submatriz local y enmascaramiento
        sub_q = q_img[rmin:rmax + 1, cmin:cmax + 1].copy()
        sub_mask = mask[rmin:rmax + 1, cmin:cmax + 1]
        sub_q[~sub_mask] = 0.0

        # Calcular QFT 2D
        F_k = qft_2d_region(sub_q)
        espectros_qft.append(F_k)

    return espectros_qft


if __name__ == "__main__":
    print("=== Prueba rápida del módulo quaternion_qft.py ===")
    
    # Imagen de prueba sintética (100x100 RGB)
    np.random.seed(42)
    img_test = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Crear dos máscaras sintéticas de superpíxeles
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[10:50, 10:50] = True

    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[50:90, 50:90] = True

    # Ejecutar canal QFT
    espectros = procesar_qft_superpixeles(img_test, [mask1, mask2])

    print(f"Número de superpíxeles procesados: {len(espectros)}")
    for i, spec in enumerate(espectros, 1):
        print(f"  Espectro {i}: dimensión {spec.shape} (M x N x 4 cuaterniones complejos)")

    print("✅ Módulo quaternion_qft.py verificado correctamente.")
