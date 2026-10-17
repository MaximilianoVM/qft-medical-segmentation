# 1. Imagen base oficial de Python ligera
FROM python:3.10-slim

# 2. Evitar archivos .pyc y forzar logs inmediatos en consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Instalar dependencias del sistema operativo para OpenCV y gráficos
RUN apt-get update &amp;&amp; apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    &amp;&amp; rm -rf /var/lib/apt/lists/*

# 4. Crear carpeta de trabajo dentro del contenedor
WORKDIR /app

# 5. Copiar requerimientos e instalar dependencias de Python
COPY environment.yml .
# Nota: Podemos usar pip install directamente sobre un requirements.txt o instalar las librerías principales
RUN pip install --no-cache-dir \
    numpy scipy scikit-learn scikit-image matplotlib Pillow typer rich joblib streamlit opencv-python pytest

# 6. Copiar el código fuente del proyecto
COPY . /app/

# 7. Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# 8. Comando por defecto al arrancar el contenedor (Lanza Streamlit)
CMD ["streamlit", "run", "src/app_streamlit.py", "--server.address=0.0.0.0"]