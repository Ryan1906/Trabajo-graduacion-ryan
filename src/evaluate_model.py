"""
Evaluación del Modelo de Reconocimiento de Emociones
=====================================================
Trabajo de Graduación - Maestría en TIC
Universidad de San Carlos de Guatemala
Facultad de Ingeniería

Autor: Ryan José Rodrigo Sigüenza Huertas

Este script evalúa el modelo emotion_model.h5 sobre el conjunto
de imágenes de prueba (data/raw/archive/Testing/) y genera las
métricas de clasificación requeridas para el Capítulo 5 de la tesis:

  - Accuracy (exactitud global)
  - Precision, Recall y F1-score por cada emoción
  - Matriz de confusión (imagen PNG)
  - Reporte de clasificación (CSV)

Las imágenes de prueba deben estar organizadas en subcarpetas,
una por emoción. El nombre de la carpeta es la etiqueta real.

Uso:
  python src/evaluate_model.py
  python src/evaluate_model.py --carpeta data/raw/archive/Testing
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)


# ====================================================================
# CONFIGURACIÓN
# ====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.h5')

DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTADOS_DIR = os.path.join(DATA_DIR, 'resultados')
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Carpeta de imágenes de prueba por defecto
TESTING_DIR_DEFAULT = os.path.join(DATA_DIR, 'raw', 'archive', 'Testing')

# IMPORTANTE: el orden de esta lista debe coincidir con el orden
# en que el modelo aprendió las clases durante el entrenamiento.
# Keras (flow_from_directory) ordena las carpetas ALFABÉTICAMENTE.
# Orden alfabético de las carpetas en inglés:
#   Angry=0, Fear=1, Happy=2, Neutral=3, Sad=4, Suprise=5
# Mapeo a español usado en el resto del proyecto:
CLASES_INGLES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Suprise"]
CLASES_ESPANOL = ["Enojado", "Miedo", "Feliz", "Neutral", "Triste", "Sorpresa"]

# Tamaño de entrada que espera el modelo
INPUT_SIZE = (48, 48)


# ====================================================================
# CARGA Y PREPARACIÓN DE IMÁGENES
# ====================================================================
def preparar_imagen(ruta_img):
    """
    Lee una imagen y la transforma al formato que espera el modelo:
    escala de grises, 48x48, normalizada y con las dimensiones (1,48,48,1).
    Devuelve None si la imagen no se pudo leer.
    """
    img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)   # dimensión de lote
    img = np.expand_dims(img, axis=-1)  # dimensión de canal
    return img


def localizar_carpeta_emociones(carpeta, profundidad_max=4):
    """
    Busca, dentro de 'carpeta', el nivel que realmente contiene las
    subcarpetas de emociones (Angry, Fear, Happy, etc.).

    Esto resuelve el caso común de carpetas anidadas como
    'Testing/Testing/Angry/...'. Devuelve la ruta correcta o, si no
    encuentra nada, la carpeta original.
    """
    nombres_validos = {n.lower() for n in CLASES_INGLES}

    def contiene_emociones(ruta):
        try:
            subs = {d.lower() for d in os.listdir(ruta)
                    if os.path.isdir(os.path.join(ruta, d))}
        except OSError:
            return False
        # Se considera válida si al menos 2 subcarpetas son emociones
        return len(subs & nombres_validos) >= 2

    # Búsqueda por niveles (BFS) hasta cierta profundidad
    pendientes = [(carpeta, 0)]
    while pendientes:
        ruta_actual, nivel = pendientes.pop(0)
        if contiene_emociones(ruta_actual):
            return ruta_actual
        if nivel >= profundidad_max:
            continue
        try:
            for d in sorted(os.listdir(ruta_actual)):
                sub = os.path.join(ruta_actual, d)
                if os.path.isdir(sub):
                    pendientes.append((sub, nivel + 1))
        except OSError:
            continue

    return carpeta  # no se encontró; se devuelve la original


def cargar_conjunto_prueba(carpeta):
    """
    Recorre las subcarpetas de 'carpeta' (una por emoción) y devuelve
    dos listas paralelas: las rutas de imagen y su etiqueta real (índice).

    La etiqueta se obtiene del nombre de la subcarpeta, comparándolo
    con CLASES_INGLES.
    """
    rutas = []
    etiquetas = []
    no_reconocidas = []

    if not os.path.isdir(carpeta):
        raise SystemExit(
            f"No se encontró la carpeta de prueba:\n  {carpeta}\n\n"
            "Verifica la ruta o pásala con --carpeta."
        )

    # Detectar automáticamente el nivel correcto (maneja Testing/Testing/...)
    carpeta_real = localizar_carpeta_emociones(carpeta)
    if carpeta_real != carpeta:
        print(f"[Info] Carpetas de emociones encontradas en: {carpeta_real}")
    carpeta = carpeta_real

    # Mapa nombre_carpeta -> índice de clase (insensible a mayúsculas)
    mapa_clases = {nombre.lower(): i for i, nombre in enumerate(CLASES_INGLES)}

    for subcarpeta in sorted(os.listdir(carpeta)):
        ruta_sub = os.path.join(carpeta, subcarpeta)
        if not os.path.isdir(ruta_sub):
            continue

        clave = subcarpeta.lower()
        if clave not in mapa_clases:
            no_reconocidas.append(subcarpeta)
            continue

        indice = mapa_clases[clave]

        # Buscar imágenes (jpg, jpeg, png)
        patrones = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        archivos = []
        for patron in patrones:
            archivos.extend(glob.glob(os.path.join(ruta_sub, patron)))

        for archivo in archivos:
            rutas.append(archivo)
            etiquetas.append(indice)

    if no_reconocidas:
        print(f"[Aviso] Subcarpetas ignoradas (no reconocidas como emoción): "
              f"{', '.join(no_reconocidas)}")

    if not rutas:
        raise SystemExit(
            "No se encontraron imágenes en las subcarpetas de prueba.\n"
            "Verifica que existan carpetas como Angry, Fear, Happy, etc."
        )

    return rutas, etiquetas


# ====================================================================
# EVALUACIÓN
# ====================================================================
def evaluar(modelo, rutas, etiquetas_reales):
    """
    Pasa cada imagen por el modelo y devuelve las predicciones
    junto con las etiquetas reales (filtrando imágenes corruptas).
    """
    y_real = []
    y_pred = []
    corruptas = 0

    total = len(rutas)
    print(f"\nEvaluando {total} imágenes de prueba…")

    for i, (ruta, etiqueta) in enumerate(zip(rutas, etiquetas_reales)):
        img = preparar_imagen(ruta)
        if img is None:
            corruptas += 1
            continue

        prediccion = modelo.predict(img, verbose=0)
        clase_predicha = int(np.argmax(prediccion))

        y_real.append(etiqueta)
        y_pred.append(clase_predicha)

        # Progreso cada 200 imágenes
        if (i + 1) % 200 == 0:
            print(f"  Procesadas {i + 1}/{total}…")

    if corruptas:
        print(f"[Aviso] {corruptas} imágenes no se pudieron leer y se omitieron.")

    print(f"Evaluación completada: {len(y_real)} imágenes válidas.\n")
    return np.array(y_real), np.array(y_pred)


# ====================================================================
# GENERACIÓN DE RESULTADOS
# ====================================================================
def generar_matriz_confusion(y_real, y_pred, salida_png):
    """Crea la matriz de confusión como imagen PNG."""
    cm = confusion_matrix(y_real, y_pred, labels=range(len(CLASES_ESPANOL)))

    # Versión normalizada (porcentajes por fila) para interpretar mejor
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Matriz de Confusión - Modelo de Reconocimiento de Emociones",
        fontsize=14, fontweight="bold"
    )

    # Matriz con conteos absolutos
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASES_ESPANOL, yticklabels=CLASES_ESPANOL, ax=ax1)
    ax1.set_title("Conteo absoluto", fontweight="bold")
    ax1.set_xlabel("Emoción predicha por el modelo")
    ax1.set_ylabel("Emoción real")

    # Matriz normalizada (porcentajes)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=CLASES_ESPANOL, yticklabels=CLASES_ESPANOL, ax=ax2)
    ax2.set_title("Proporción por emoción real (la diagonal = aciertos)",
                  fontweight="bold")
    ax2.set_xlabel("Emoción predicha por el modelo")
    ax2.set_ylabel("Emoción real")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(salida_png, dpi=150, bbox_inches="tight")
    print(f"[OK] Matriz de confusión guardada en: {salida_png}")
    plt.close(fig)
    return cm


def generar_reporte_metricas(y_real, y_pred, salida_csv):
    """Genera el reporte de clasificación (precision, recall, F1) en CSV."""
    reporte = classification_report(
        y_real, y_pred,
        labels=range(len(CLASES_ESPANOL)),
        target_names=CLASES_ESPANOL,
        output_dict=True,
        zero_division=0,
    )

    filas = []
    for emocion in CLASES_ESPANOL:
        m = reporte.get(emocion, {})
        filas.append({
            "emocion": emocion,
            "precision": round(m.get("precision", 0.0), 4),
            "recall": round(m.get("recall", 0.0), 4),
            "f1_score": round(m.get("f1-score", 0.0), 4),
            "muestras": int(m.get("support", 0)),
        })

    df = pd.DataFrame(filas)
    df.to_csv(salida_csv, index=False)
    print(f"[OK] Reporte de métricas guardado en: {salida_csv}")
    return df, reporte


def generar_grafica_metricas(df_metricas, salida_png):
    """Crea una gráfica de barras con precision, recall y F1 por emoción."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df_metricas))
    ancho = 0.25

    ax.bar(x - ancho, df_metricas["precision"], ancho,
           label="Precisión", color="#3498db")
    ax.bar(x, df_metricas["recall"], ancho,
           label="Recall", color="#2ecc71")
    ax.bar(x + ancho, df_metricas["f1_score"], ancho,
           label="F1-score", color="#e67e22")

    ax.set_xlabel("Emoción")
    ax.set_ylabel("Valor de la métrica (0 a 1)")
    ax.set_title("Métricas de Clasificación por Emoción",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(df_metricas["emocion"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(salida_png, dpi=150, bbox_inches="tight")
    print(f"[OK] Gráfica de métricas guardada en: {salida_png}")
    plt.close(fig)


# ====================================================================
# MAIN
# ====================================================================
def main():
    args = sys.argv[1:]

    # Carpeta de prueba (por defecto o pasada con --carpeta)
    carpeta = TESTING_DIR_DEFAULT
    if "--carpeta" in args:
        idx = args.index("--carpeta")
        if idx + 1 < len(args):
            ruta = args[idx + 1]
            carpeta = ruta if os.path.isabs(ruta) else os.path.join(BASE_DIR, ruta)

    print("=" * 64)
    print("EVALUACIÓN DEL MODELO DE RECONOCIMIENTO DE EMOCIONES")
    print("=" * 64)

    # 1. Cargar el modelo
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No se encontró el modelo en: {MODEL_PATH}")
    print(f"Cargando modelo: {MODEL_PATH}")
    modelo = load_model(MODEL_PATH, compile=False)

    # Verificación del número de clases
    n_salidas = modelo.output_shape[-1]
    print(f"El modelo produce {n_salidas} clases de salida.")
    if n_salidas != len(CLASES_ESPANOL):
        print(f"[ADVERTENCIA] El modelo tiene {n_salidas} clases, pero el "
              f"script espera {len(CLASES_ESPANOL)}. Revisa el orden de las "
              f"clases en CLASES_INGLES / CLASES_ESPANOL.")

    print(f"Orden de clases asumido: "
          f"{', '.join(f'{i}={c}' for i, c in enumerate(CLASES_ESPANOL))}")

    # 2. Cargar el conjunto de prueba
    print(f"\nCarpeta de prueba: {carpeta}")
    rutas, etiquetas = cargar_conjunto_prueba(carpeta)
    print(f"Imágenes de prueba encontradas: {len(rutas)}")

    # Distribución por clase
    conteo = pd.Series(etiquetas).value_counts().sort_index()
    print("Distribución por emoción:")
    for idx, n in conteo.items():
        print(f"  {CLASES_ESPANOL[idx]:<10} {n} imágenes")

    # 3. Evaluar
    y_real, y_pred = evaluar(modelo, rutas, etiquetas)

    # 4. Calcular métricas globales
    accuracy = accuracy_score(y_real, y_pred)
    f1_macro = f1_score(y_real, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_real, y_pred, average="weighted", zero_division=0)

    # 5. Generar salidas
    salida_cm = os.path.join(RESULTADOS_DIR, "matriz_confusion.png")
    salida_csv = os.path.join(RESULTADOS_DIR, "metricas_modelo.csv")
    salida_barras = os.path.join(RESULTADOS_DIR, "metricas_por_emocion.png")

    generar_matriz_confusion(y_real, y_pred, salida_cm)
    df_metricas, _ = generar_reporte_metricas(y_real, y_pred, salida_csv)
    generar_grafica_metricas(df_metricas, salida_barras)

    # 6. Resumen en consola
    print("\n" + "=" * 64)
    print("RESULTADOS DE LA EVALUACIÓN")
    print("=" * 64)
    print(f"  Exactitud global (accuracy):   {accuracy * 100:.2f}%")
    print(f"  F1-score promedio (macro):     {f1_macro:.4f}")
    print(f"  F1-score ponderado (weighted): {f1_weighted:.4f}")
    print()
    print("  Métricas por emoción:")
    print("  " + "-" * 56)
    print(f"  {'Emoción':<10} {'Precisión':>10} {'Recall':>10} "
          f"{'F1-score':>10} {'Muestras':>10}")
    print("  " + "-" * 56)
    for _, fila in df_metricas.iterrows():
        print(f"  {fila['emocion']:<10} {fila['precision']:>10.4f} "
              f"{fila['recall']:>10.4f} {fila['f1_score']:>10.4f} "
              f"{fila['muestras']:>10}")
    print("  " + "-" * 56)

    print("\n" + "=" * 64)
    print("PROCESO COMPLETADO. Archivos generados en data/resultados/:")
    print(f"  - {os.path.basename(salida_cm)}")
    print(f"  - {os.path.basename(salida_csv)}")
    print(f"  - {os.path.basename(salida_barras)}")
    print("=" * 64)


if __name__ == "__main__":
    main()