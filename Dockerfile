FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workdir

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    scikit-image \
    scikit-learn \
    opencv-python-headless \
    joblib \
    typer \
    rich \
    streamlit \
    pytest \
    flake8 \
    black \
    jupyterlab

COPY . .

EXPOSE 8080 8501

CMD ["bash"]