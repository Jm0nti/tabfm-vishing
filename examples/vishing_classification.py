# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Clasificacion de sesiones de vishing con TabFM v1.0.0.

Adapta el ejemplo `classification_example.py` al dataset sintetico de BioCatch
(`data/biocatch_sinthetic_data.csv`) para predecir si una sesion de un usuario
dentro de la app bancaria corresponde a un caso de vishing (`is_vishing`).

Consideraciones de diseno:

- TabFM usa *in-context learning*: no ajusta pesos, sino que "lee" el set de
  entrenamiento como contexto al momento de predecir. El costo crece con el
  numero de filas de contexto, por lo que en CPU se submuestrea un contexto
  estratificado de unos pocos miles de filas (ver CONTEXT_SIZE).

- Se construye un modelo de "senal propia": se EXCLUYEN las columnas del
  proveedor (`biocatch_*`) para no copiar su respuesta, ademas de los
  identificadores y la metadata posterior al reclamo (`days_to_claim`,
  `claim_category`), que serian data leakage.

- El dataset esta muy desbalanceado (~5% vishing), asi que se reporta PR-AUC,
  ROC-AUC, matriz de confusion y un barrido de umbrales en lugar de accuracy.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import tabfm


# --- Configuracion -----------------------------------------------------------

# Ruta al CSV (relativa a este archivo: tabfm/examples/ -> tabfm/data/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_THIS_DIR, "..", "data", "biocatch_sinthetic_data.csv")

TARGET = "is_vishing"

# Columnas excluidas del feature set.
ID_COLS = ["session_id", "customer_id", "session_timestamp"]
LEAKAGE_COLS = ["days_to_claim", "claim_category"]  # metadata posterior al reclamo
BIOCATCH_COLS = [
    "biocatch_risk_score",
    "biocatch_genuine_score",
    "biocatch_ato_indicator",
    "biocatch_social_eng_indicator",
    "biocatch_bot_indicator",
]

# Tamano del contexto (train) que se pasa a TabFM en inferencia y del hold-out
# de evaluacion. Ambos se muestrean de forma estratificada preservando la
# proporcion de clases (~5% vishing). Reducir si la ejecucion es muy lenta.
CONTEXT_SIZE = 4000
TEST_SIZE = 2000

# n_estimators es el principal control de velocidad/calidad en CPU: el default
# de la libreria es 32; se baja para que la POC corra en minutos.
N_ESTIMATORS = 8

RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
  """Carga el CSV y separa features (X) del objetivo (y)."""
  df = pd.read_csv(DATA_PATH)

  drop_cols = ID_COLS + LEAKAGE_COLS + BIOCATCH_COLS + [TARGET]
  feature_cols = [c for c in df.columns if c not in drop_cols]

  X = df[feature_cols].copy()
  y = df[TARGET].astype(int)
  return X, y


def stratified_sample(
    X: pd.DataFrame, y: pd.Series, n: int
) -> tuple[pd.DataFrame, pd.Series]:
  """Toma una muestra estratificada de tamano `n` (o todo X si n >= len(X))."""
  if n >= len(X):
    return X, y
  X_sample, _, y_sample, _ = train_test_split(
      X, y, train_size=n, stratify=y, random_state=RANDOM_STATE
  )
  return X_sample, y_sample


def evaluate(y_true: np.ndarray, proba_vishing: np.ndarray) -> None:
  """Imprime metricas apropiadas para un problema desbalanceado."""
  roc = roc_auc_score(y_true, proba_vishing)
  pr_auc = average_precision_score(y_true, proba_vishing)

  print("\n=== Metricas globales (independientes del umbral) ===")
  print(f"ROC-AUC          : {roc:.4f}")
  print(f"PR-AUC (avg prec): {pr_auc:.4f}   (baseline = prevalencia = "
        f"{y_true.mean():.4f})")

  print("\n=== Barrido de umbrales (clase positiva = vishing) ===")
  print(f"{'umbral':>7} | {'precision':>9} | {'recall':>7} | {'f1':>6} | "
        f"{'flagged':>7}")
  for thr in (0.3, 0.5, 0.7):
    y_pred = (proba_vishing >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"{thr:>7.2f} | {p:>9.3f} | {r:>7.3f} | {f1:>6.3f} | "
          f"{int(y_pred.sum()):>7}")

  print("\n=== Reporte de clasificacion (umbral 0.5) ===")
  y_pred_05 = (proba_vishing >= 0.5).astype(int)
  print(classification_report(
      y_true, y_pred_05, target_names=["legitima", "vishing"], digits=3,
      zero_division=0,
  ))
  print("Matriz de confusion (filas=real, cols=pred) [legitima, vishing]:")
  print(confusion_matrix(y_true, y_pred_05))


def run(model=None) -> None:
  """Carga datos, ajusta el contexto de TabFM y evalua en el hold-out."""
  if model is None:
    # Backend JAX (default). Para PyTorch:
    # model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")

  X, y = load_dataset()
  print(f"Dataset: {len(X)} sesiones, {X.shape[1]} features, "
        f"prevalencia vishing = {y.mean():.4f}")
  print(f"Features usadas: {list(X.columns)}")

  # 1. Hold-out estratificado para evaluacion.
  X_pool, X_test, y_pool, y_test = train_test_split(
      X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
  )

  # 2. Contexto estratificado (lo que TabFM "lee" en inferencia).
  X_ctx, y_ctx = stratified_sample(X_pool, y_pool, CONTEXT_SIZE)
  print(f"\nContexto (train): {len(X_ctx)} filas "
        f"({int(y_ctx.sum())} vishing) | Test: {len(X_test)} filas "
        f"({int(y_test.sum())} vishing)")

  # 3. Clasificador compatible con scikit-learn.
  clf = tabfm.TabFMClassifier(
      model=model, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
  )

  # 4. "Fit" = preparar transformadores y fijar el contexto.
  t0 = time.time()
  clf.fit(X_ctx, y_ctx)
  print(f"fit() completado en {time.time() - t0:.1f}s")

  # 5. Probabilidades sobre el test. Se toma la columna de la clase 1 (vishing).
  t0 = time.time()
  proba = clf.predict_proba(X_test)
  print(f"predict_proba() completado en {time.time() - t0:.1f}s")

  vishing_idx = list(clf.classes_).index(1)
  proba_vishing = proba[:, vishing_idx]

  # 6. Evaluacion.
  evaluate(y_test.to_numpy(), proba_vishing)


if __name__ == "__main__":
  print("Ejecutando clasificacion de vishing con TabFM... "
        "(la compilacion y la primera inferencia pueden tardar varios minutos)")
  run()
