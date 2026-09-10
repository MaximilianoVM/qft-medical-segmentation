"""
Implementacion de SLIC (Simple Linear Iterative Clustering)
siguiendo el Algoritmo 1 y la ecuacion (3) de:

Achanta et al., "SLIC Superpixels Compared to State-of-the-Art
Superpixel Methods", IEEE TPAMI 2012.

Distancia (ec. 3):
    D = sqrt( dc^2 + (ds/S)^2 * m^2 )

donde:
    dc = distancia euclidiana en color CIELAB entre pixel y centro
    ds = distancia euclidiana espacial (x,y) entre pixel y centro
    S  = sqrt(N/k), intervalo de muestreo de la rejilla
    m  = compacidad (controla peso relativo color vs. espacio)
"""

from tabnanny import verbose

import numpy as np
from skimage import data
from skimage.color import rgb2lab


def init_cluster_centers(lab_image, S):
    """Paso de inicializacion: siembra centros en una rejilla regular
    separados por S pixeles, y los mueve a la posicion de menor
    gradiente en una vecindad 3x3 (Seccion 3.1)."""
    H, W, _ = lab_image.shape

    # Gradiente simple (magnitud) usado solo para reubicar semillas
    gy, gx = np.gradient(lab_image[:, :, 0])  # se usa canal L
    grad_mag = gx ** 2 + gy ** 2

    centers = []
    for y in range(S // 2, H, S):
        for x in range(S // 2, W, S):
            # Buscar el minimo de gradiente en una vecindad 3x3
            y0, y1 = max(0, y - 1), min(H, y + 2)
            x0, x1 = max(0, x - 1), min(W, x + 2)
            neighborhood = grad_mag[y0:y1, x0:x1]
            iy, ix = np.unravel_index(np.argmin(neighborhood), neighborhood.shape)
            ny, nx = y0 + iy, x0 + ix

            l, a, b = lab_image[ny, nx]
            centers.append([l, a, b, nx, ny])  # [l, a, b, x, y]

    return np.array(centers, dtype=np.float64)


def slic(image_rgb, k, m=10, num_iters=10):
    """
    image_rgb : array HxWx3, valores en [0,1] o [0,255]
    k         : numero deseado de superpixeles
    m         : compacidad (rango sugerido por el paper: [1, 40])
    num_iters : iteraciones de asignacion/actualizacion (el paper usa 10)

    Retorna: labels (HxW) con el indice de cluster de cada pixel.
    """
    H, W, _ = image_rgb.shape
    N = H * W
    S = int(np.sqrt(N / k))  # intervalo de la rejilla

    lab_image = rgb2lab(image_rgb)

    centers = init_cluster_centers(lab_image, S)
    n_clusters = centers.shape[0]

    labels = -1 * np.ones((H, W), dtype=np.int64)
    distances = np.full((H, W), np.inf)

    # Coordenadas de pixel precomputadas
    yy, xx = np.mgrid[0:H, 0:W]

    for iteration in range(num_iters):
        distances[:] = np.inf
        labels[:] = -1

        for k_idx in range(n_clusters):
            l_c, a_c, b_c, x_c, y_c = centers[k_idx]

            # Region de busqueda 2S x 2S alrededor del centro (clave de SLIC)
            y0, y1 = int(max(0, y_c - S)), int(min(H, y_c + S + 1))
            x0, x1 = int(max(0, x_c - S)), int(min(W, x_c + S + 1))

            patch_lab = lab_image[y0:y1, x0:x1]
            patch_y = yy[y0:y1, x0:x1]
            patch_x = xx[y0:y1, x0:x1]

            dc = np.sqrt(
                (patch_lab[:, :, 0] - l_c) ** 2
                + (patch_lab[:, :, 1] - a_c) ** 2
                + (patch_lab[:, :, 2] - b_c) ** 2
            )
            ds = np.sqrt((patch_x - x_c) ** 2 + (patch_y - y_c) ** 2)

            D = np.sqrt(dc ** 2 + (ds / S) ** 2 * m ** 2)  # ecuacion (3)

            patch_dist = distances[y0:y1, x0:x1]
            patch_labels = labels[y0:y1, x0:x1]

            mask = D < patch_dist
            patch_dist[mask] = D[mask]
            patch_labels[mask] = k_idx

        # Paso de actualizacion: nuevo centro = media [l a b x y] del cluster
        new_centers = np.zeros_like(centers)
        counts = np.zeros(n_clusters)
        for k_idx in range(n_clusters):
            mask = labels == k_idx
            if not np.any(mask):
                new_centers[k_idx] = centers[k_idx]  # cluster vacio: se conserva
                continue
            l_mean = lab_image[:, :, 0][mask].mean()
            a_mean = lab_image[:, :, 1][mask].mean()
            b_mean = lab_image[:, :, 2][mask].mean()
            x_mean = xx[mask].mean()
            y_mean = yy[mask].mean()
            new_centers[k_idx] = [l_mean, a_mean, b_mean, x_mean, y_mean]

        # Error residual E (norma L2 entre centros nuevos y anteriores)
        E = np.sqrt(np.sum((new_centers[:, :2] - centers[:, :2]) ** 2))
        centers = new_centers

        if verbose:
            print(f"Iteracion {iteration + 1}/{num_iters} - error residual E = {E:.4f}")

    labels = enforce_connectivity(labels, S)
    return labels


def enforce_connectivity(labels, S):
    """Postprocesamiento simple (Seccion 3.3): reasigna pixeles
    'huerfanos' (que no forman una componente conexa con su centro)
    a la etiqueta de un vecino ya procesado."""
    H, W = labels.shape
    new_labels = -1 * np.ones_like(labels)
    label_counter = 0
    min_size = (S * S) // 4

    for y in range(H):
        for x in range(W):
            if new_labels[y, x] != -1:
                continue

            old_label = labels[y, x]
            # Flood fill (BFS) de la componente conexa que contiene (y, x)
            component = [(y, x)]
            new_labels[y, x] = label_counter
            queue = [(y, x)]
            adjacent_label = old_label

            while queue:
                cy, cx = queue.pop(0)  # BFS con lista simple
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and new_labels[ny, nx] == -1:
                        if labels[ny, nx] == old_label:
                            new_labels[ny, nx] = label_counter
                            component.append((ny, nx))
                            queue.append((ny, nx))
                        else:
                            adjacent_label = new_labels[ny, nx] if new_labels[ny, nx] != -1 else adjacent_label

            if len(component) < min_size and label_counter > 0:
                # Componente demasiado pequena: se fusiona con un vecino
                for (cy, cx) in component:
                    new_labels[cy, cx] = adjacent_label
            else:
                label_counter += 1

    return new_labels




