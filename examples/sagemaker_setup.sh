#!/usr/bin/env bash
# =============================================================================
# Setup de entorno para correr TabFM (backend JAX + GPU) en SageMaker.
#
# Uso (desde la Terminal de Jupyter en la Notebook Instance):
#   bash examples/sagemaker_setup.sh
#
# Requisitos:
#   - Instancia con GPU NVIDIA (p. ej. ml.g4dn.xlarge). El driver ya viene
#     preinstalado; jax[cuda12] trae las librerias CUDA 12 via pip (no hace
#     falta instalar CUDA a nivel de sistema).
#   - El repo `tabfm` subido a la instancia (ver variable REPO_DIR abajo).
# =============================================================================
set -euo pipefail

# --- Ajusta esto a donde subiste el repo (carpeta que contiene pyproject.toml) ---
REPO_DIR="${REPO_DIR:-$HOME/SageMaker/tabfm}"
ENV_NAME="${ENV_NAME:-tabfm-gpu}"

echo ">>> 0. GPU visible en la instancia:"
nvidia-smi || { echo "!! No se detecta GPU. Usa una instancia con GPU."; exit 1; }

echo ">>> 1. Creando entorno conda '${ENV_NAME}' (Python 3.11)..."
# En SageMaker Notebook Instances 'conda' ya existe.
conda create -y -n "${ENV_NAME}" python=3.11
# 'conda activate' necesita cargar la funcion de shell:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo ">>> 2. Instalando TabFM con backend JAX + soporte CUDA 12..."
cd "${REPO_DIR}"
python -m pip install --upgrade pip
# Backend JAX (CPU) + core de tabfm:
python -m pip install -e ".[jax]"
# Reemplaza jaxlib CPU por el plugin CUDA 12 (misma version que el jax que funciona en local).
# Si prefieres la ultima compatible, usa simplemente:  pip install -e ".[jax,cuda]"
python -m pip install --upgrade "jax[cuda12]==0.10.2"

echo ">>> 3. Registrando kernel de Jupyter para el notebook..."
python -m pip install ipykernel
python -m ipykernel install --user --name "${ENV_NAME}" \
  --display-name "Python (${ENV_NAME})"

echo ">>> 4. Verificando que JAX ve la GPU..."
python - <<'PY'
import jax
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
assert any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in jax.devices()), \
    "JAX NO ve la GPU. Revisa la instalacion de jax[cuda12] y el driver (nvidia-smi)."
print("OK: JAX corriendo sobre GPU.")
PY

echo ""
echo ">>> LISTO."
echo "    1. Abre examples/vishing_classification.ipynb"
echo "    2. Selecciona el kernel: 'Python (${ENV_NAME})'"
echo "    3. Ejecuta las celdas de arriba hacia abajo."
echo ""
echo "    Nota: la primera ejecucion descarga los pesos del modelo (~1.5 GB, 134"
echo "    archivos) a ~/.cache/huggingface. Asegurate de tener volumen EBS >= 25 GB"
echo "    en la Notebook Instance."
