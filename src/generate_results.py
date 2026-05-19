"""
Generación de Resultados - Prototipo de Análisis Emocional
===========================================================
Trabajo de Graduación - Maestría en TIC
Universidad de San Carlos de Guatemala
Facultad de Ingeniería

Autor: Ryan José Rodrigo Sigüenza Huertas

Procesa los CSV de sesión generados por real_time_interface.py
y produce una imagen PNG con 4 gráficas + un CSV resumen.

Estructura de carpetas:
  data/sesiones/    -> CSV de entrada (generados por la interfaz)
  data/resultados/  -> PNG y CSV resumen (salida de este script)

Reglas de nombrado de la salida:
  - 1 archivo procesado  -> usa el nombre de esa sesión
  - 2 o más archivos     -> usa un nombre GENERAL con fecha
                            (resultado_general_AAAAMMDD_HHMMSS)

Las gráficas están alineadas con los objetivos del prototipo:
  OE1: Influencia de las emociones en la participación.
  OE2: Retroalimentación emocional para decisiones pedagógicas.
  OE3: Visualización de emociones (interfaz / salida gráfica).
  OE4: Implicaciones técnicas: precisión y usabilidad de la detección.

Uso:
  python src/generate_results.py                  # todas las sesiones
  python src/generate_results.py --ultima         # solo la más reciente
  python src/generate_results.py data/sesiones/sesion_x.csv [otra.csv ...]
"""

import os
import sys
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ====================================================================
# CONFIGURACIÓN Y ESTRUCTURA DE CARPETAS
# ====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SESIONES_DIR = os.path.join(DATA_DIR, 'sesiones')      # CSV de entrada
RESULTADOS_DIR = os.path.join(DATA_DIR, 'resultados')  # PNG + CSV de salida

for carpeta in (DATA_DIR, SESIONES_DIR, RESULTADOS_DIR):
    os.makedirs(carpeta, exist_ok=True)

EMOCIONES = ["Enojado", "Miedo", "Feliz", "Neutral", "Triste", "Sorpresa"]

# Clasificación pedagógica (OE1 y OE2)
EMOCIONES_POSITIVAS = ["Feliz", "Sorpresa"]
EMOCIONES_NEGATIVAS = ["Enojado", "Miedo", "Triste"]
EMOCIONES_NEUTRAS = ["Neutral"]

COLORES_EMOCION = {
    "Enojado": "#e74c3c",
    "Miedo": "#9b59b6",
    "Feliz": "#2ecc71",
    "Neutral": "#95a5a6",
    "Triste": "#3498db",
    "Sorpresa": "#f1c40f",
}

TRADUCCION_EN_ES = {
    "Angry": "Enojado",
    "Fear": "Miedo",
    "Happy": "Feliz",
    "Neutral": "Neutral",
    "Sad": "Triste",
    "Surprise": "Sorpresa",
    "Suprise": "Sorpresa",
}


# ====================================================================
# CARGA Y NORMALIZACIÓN DE DATOS
# ====================================================================
def normalizar_dataframe(df):
    """Acepta el formato NUEVO o el VIEJO y devuelve un DataFrame estándar."""
    df = df.copy()
    columnas = set(df.columns)

    if {"tiempo_seg", "emocion_grupal"}.issubset(columnas):
        if "estudiantes_detectados" not in df.columns:
            df["estudiantes_detectados"] = 1
        if "sesion" not in df.columns:
            df["sesion"] = "sesion_unica"
        return df

    if {"segundo", "emocion"}.issubset(columnas):
        df = df.rename(columns={
            "segundo": "tiempo_seg",
            "emocion": "emocion_grupal",
        })
        df["emocion_grupal"] = df["emocion_grupal"].map(
            lambda e: TRADUCCION_EN_ES.get(e, e)
        )
        if "estudiantes_detectados" not in df.columns:
            df["estudiantes_detectados"] = 1
        df["sesion"] = "sesion_legado"
        return df

    raise ValueError(
        "El CSV no tiene el formato esperado.\n"
        f"Columnas encontradas: {sorted(columnas)}\n"
        "Se esperaba ('tiempo_seg','emocion_grupal') o ('segundo','emocion')."
    )


def cargar_datos(rutas):
    """Carga uno o varios CSV y los combina en un solo DataFrame."""
    frames = []
    for ruta in rutas:
        if not os.path.exists(ruta):
            print(f"[Aviso] No se encontró: {ruta}")
            continue
        try:
            df = pd.read_csv(ruta)
            df = normalizar_dataframe(df)
            df["archivo_origen"] = os.path.basename(ruta)
            frames.append(df)
            print(f"[OK] Cargado: {os.path.basename(ruta)}  ({len(df)} registros)")
        except Exception as e:
            print(f"[Error] No se pudo procesar {ruta}: {e}")

    if not frames:
        raise SystemExit(
            "No se cargó ningún CSV válido. Genera primero una sesión "
            "con real_time_interface.py."
        )

    return pd.concat(frames, ignore_index=True)


# ====================================================================
# CÁLCULO DE ESTADÍSTICAS
# ====================================================================
def calcular_estadisticas(df):
    """Devuelve un diccionario con las métricas clave."""
    total = len(df)

    conteo = df["emocion_grupal"].value_counts()
    distribucion = (conteo / total * 100).round(2)

    def clasifica(emo):
        if emo in EMOCIONES_POSITIVAS:
            return "Positiva"
        if emo in EMOCIONES_NEGATIVAS:
            return "Negativa"
        return "Neutra"

    df = df.copy()
    df["categoria"] = df["emocion_grupal"].apply(clasifica)
    cat_conteo = df["categoria"].value_counts()
    cat_pct = (cat_conteo / total * 100).round(2)

    # Duración: si hay varias sesiones, sumar la duración de cada una
    if "archivo_origen" in df.columns:
        duracion = 0.0
        for _, grupo in df.groupby("archivo_origen"):
            duracion += grupo["tiempo_seg"].max() - grupo["tiempo_seg"].min()
    else:
        duracion = df["tiempo_seg"].max() - df["tiempo_seg"].min()

    prom_estudiantes = df["estudiantes_detectados"].mean()
    max_estudiantes = df["estudiantes_detectados"].max()
    emocion_dominante = conteo.idxmax()

    # Número de sesiones distintas analizadas
    n_sesiones = df["archivo_origen"].nunique() if "archivo_origen" in df.columns else 1

    return {
        "total_registros": total,
        "duracion_seg": round(duracion, 1),
        "conteo": conteo,
        "distribucion_pct": distribucion,
        "categoria_conteo": cat_conteo,
        "categoria_pct": cat_pct,
        "prom_estudiantes": round(prom_estudiantes, 2),
        "max_estudiantes": int(max_estudiantes),
        "emocion_dominante": emocion_dominante,
        "n_sesiones": n_sesiones,
        "df_categorizado": df,
    }


# ====================================================================
# GENERACIÓN DE GRÁFICAS
# ====================================================================
def generar_graficas(df, stats, salida_png, titulo_extra=""):
    """Crea la figura con 4 paneles y la guarda en disco."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    titulo = "Resultados del Prototipo de Análisis Emocional en Videollamadas Educativas"
    if titulo_extra:
        titulo += f"\n{titulo_extra}"
    fig.suptitle(titulo, fontsize=14, fontweight="bold", y=0.99)
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # GRÁFICA 1 (OE3): Evolución emocional en el tiempo
    emocion_to_num = {em: i for i, em in enumerate(EMOCIONES)}
    df_plot = df.copy()
    df_plot["emocion_num"] = df_plot["emocion_grupal"].map(emocion_to_num)
    df_plot = df_plot.dropna(subset=["emocion_num"]).sort_values("tiempo_seg")
    colores_puntos = df_plot["emocion_grupal"].map(COLORES_EMOCION)

    ax1.plot(df_plot["tiempo_seg"], df_plot["emocion_num"],
             color="#bdc3c7", linewidth=1, zorder=1)
    ax1.scatter(df_plot["tiempo_seg"], df_plot["emocion_num"],
                c=colores_puntos, s=45, zorder=2, edgecolors="white", linewidth=0.5)
    ax1.set_yticks(range(len(EMOCIONES)))
    ax1.set_yticklabels(EMOCIONES)
    ax1.set_title("OE3 · Evolución emocional durante la sesión",
                  fontweight="bold")
    ax1.set_xlabel("Tiempo de grabación (segundos)")
    ax1.set_ylabel("Emoción grupal detectada")

    # GRÁFICA 2: Distribución total (barras)
    conteo = stats["conteo"].reindex(EMOCIONES).fillna(0)
    colores_barras = [COLORES_EMOCION[e] for e in conteo.index]
    barras = ax2.bar(conteo.index, conteo.values, color=colores_barras,
                     edgecolor="white")
    ax2.set_title("Distribución total de estados emocionales",
                  fontweight="bold")
    ax2.set_xlabel("Emoción")
    ax2.set_ylabel("Frecuencia (cantidad de registros)")
    ax2.tick_params(axis="x", rotation=20)
    total = stats["total_registros"]
    for barra, valor in zip(barras, conteo.values):
        pct = valor / total * 100 if total else 0
        ax2.text(barra.get_x() + barra.get_width() / 2,
                 barra.get_height() + total * 0.01,
                 f"{pct:.1f}%", ha="center", fontsize=9)

    # GRÁFICA 3 (OE1): Clima emocional
    cat_pct = stats["categoria_pct"].reindex(
        ["Positiva", "Neutra", "Negativa"]
    ).fillna(0)
    colores_cat = {"Positiva": "#2ecc71", "Neutra": "#95a5a6", "Negativa": "#e74c3c"}
    ax3.pie(
        cat_pct.values,
        labels=cat_pct.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=[colores_cat[c] for c in cat_pct.index],
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    ax3.set_title(
        "OE1 · Clima emocional del grupo\n"
        "(emociones que favorecen vs. dificultan la participación)",
        fontweight="bold"
    )

    # GRÁFICA 4 (OE2): Momentos de intervención docente
    df_cat = stats["df_categorizado"].sort_values("tiempo_seg")
    señal_negativa = (df_cat["categoria"] == "Negativa").astype(int)
    ax4.fill_between(df_cat["tiempo_seg"], señal_negativa,
                     step="mid", color="#e74c3c", alpha=0.6,
                     label="Momento de alerta (emoción negativa)")
    ax4.plot(df_cat["tiempo_seg"], señal_negativa, drawstyle="steps-mid",
             color="#c0392b", linewidth=1)
    ax4.set_ylim(-0.1, 1.2)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["Estable", "Alerta"])
    ax4.set_title("OE2 · Momentos sugeridos de intervención docente",
                  fontweight="bold")
    ax4.set_xlabel("Tiempo de grabación (segundos)")
    ax4.set_ylabel("Estado del grupo")
    ax4.legend(loc="upper right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(salida_png, dpi=150, bbox_inches="tight")
    print(f"[OK] Gráficas guardadas en: {salida_png}")
    plt.close(fig)


# ====================================================================
# CSV RESUMEN
# ====================================================================
def guardar_resumen_csv(stats, salida_csv):
    """Guarda un CSV resumen con distribución de emociones y métricas generales."""
    filas_emocion = []
    for emo in EMOCIONES:
        if emo in EMOCIONES_POSITIVAS:
            categoria = "Positiva"
        elif emo in EMOCIONES_NEGATIVAS:
            categoria = "Negativa"
        else:
            categoria = "Neutra"
        filas_emocion.append({
            "metrica": f"emocion_{emo}",
            "categoria": categoria,
            "frecuencia": int(stats["conteo"].get(emo, 0)),
            "porcentaje": float(stats["distribucion_pct"].get(emo, 0.0)),
        })

    metricas_generales = [
        {"metrica": "sesiones_analizadas", "categoria": "General",
         "frecuencia": stats["n_sesiones"], "porcentaje": ""},
        {"metrica": "total_registros", "categoria": "General",
         "frecuencia": stats["total_registros"], "porcentaje": ""},
        {"metrica": "duracion_segundos", "categoria": "General",
         "frecuencia": stats["duracion_seg"], "porcentaje": ""},
        {"metrica": "emocion_predominante", "categoria": "General",
         "frecuencia": stats["emocion_dominante"], "porcentaje": ""},
        {"metrica": "estudiantes_promedio", "categoria": "General",
         "frecuencia": stats["prom_estudiantes"], "porcentaje": ""},
        {"metrica": "estudiantes_maximo", "categoria": "General",
         "frecuencia": stats["max_estudiantes"], "porcentaje": ""},
        {"metrica": "clima_positivo", "categoria": "Clima",
         "frecuencia": int(stats["categoria_conteo"].get("Positiva", 0)),
         "porcentaje": float(stats["categoria_pct"].get("Positiva", 0.0))},
        {"metrica": "clima_neutro", "categoria": "Clima",
         "frecuencia": int(stats["categoria_conteo"].get("Neutra", 0)),
         "porcentaje": float(stats["categoria_pct"].get("Neutra", 0.0))},
        {"metrica": "clima_negativo", "categoria": "Clima",
         "frecuencia": int(stats["categoria_conteo"].get("Negativa", 0)),
         "porcentaje": float(stats["categoria_pct"].get("Negativa", 0.0))},
    ]

    resumen = pd.DataFrame(filas_emocion + metricas_generales)
    resumen.to_csv(salida_csv, index=False)
    print(f"[OK] Resumen estadístico guardado en: {salida_csv}")
    return resumen


# ====================================================================
# RESOLUCIÓN DE RUTAS Y NOMBRE DE SALIDA
# ====================================================================
def resolver_rutas(args):
    """
    Determina qué CSV procesar.
    Por defecto procesa TODAS las sesiones de data/sesiones/.
    """
    # --ultima: solo la sesión más reciente
    if "--ultima" in args:
        rutas = sorted(glob.glob(os.path.join(SESIONES_DIR, "sesion_*.csv")))
        return [rutas[-1]] if rutas else []

    # Rutas explícitas pasadas por el usuario
    rutas_explicitas = [a for a in args if a.endswith(".csv")]
    if rutas_explicitas:
        resueltas = []
        for r in rutas_explicitas:
            resueltas.append(r if os.path.isabs(r) else os.path.join(BASE_DIR, r))
        return resueltas

    # Por defecto: TODAS las sesiones
    rutas = sorted(glob.glob(os.path.join(SESIONES_DIR, "sesion_*.csv")))
    return rutas


def construir_nombre_base(rutas):
    """
    Decide el nombre base de los archivos de salida.

    - 1 archivo  -> usa el nombre de esa sesión (sin la extensión)
    - 2 o más    -> nombre GENERAL con fecha, para no tomar
                    el nombre de un archivo cualquiera de la lista.
    """
    if len(rutas) == 1:
        return os.path.splitext(os.path.basename(rutas[0]))[0]

    # Varios archivos: nombre general con marca de tiempo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"resultado_general_{timestamp}"


# ====================================================================
# MAIN
# ====================================================================
def main():
    args = sys.argv[1:]
    rutas = resolver_rutas(args)

    if not rutas:
        print(
            "No se encontró ningún CSV de sesión para procesar.\n"
            "Genera primero una sesión con:\n"
            "  python src/real_time_interface.py\n\n"
            "Los CSV deben estar en: data/sesiones/"
        )
        sys.exit(1)

    print("=" * 60)
    print("GENERANDO RESULTADOS DEL PROTOTIPO")
    print("=" * 60)
    print(f"Archivos a procesar: {len(rutas)}")

    df = cargar_datos(rutas)
    stats = calcular_estadisticas(df)

    # Nombre base (general si son varios archivos)
    base = construir_nombre_base(rutas)

    salida_png = os.path.join(RESULTADOS_DIR, f"grafica_{base}.png")
    salida_csv = os.path.join(RESULTADOS_DIR, f"resumen_{base}.csv")

    # Subtítulo de la gráfica según cuántas sesiones
    if len(rutas) == 1:
        titulo_extra = f"Sesión: {os.path.basename(rutas[0])}"
    else:
        titulo_extra = (f"Análisis consolidado de {len(rutas)} sesiones  ·  "
                        f"{datetime.now().strftime('%d/%m/%Y')}")

    generar_graficas(df, stats, salida_png, titulo_extra)
    guardar_resumen_csv(stats, salida_csv)

    # Resumen en consola
    print("\n" + "-" * 60)
    print("RESUMEN")
    print("-" * 60)
    print(f"  Sesiones analizadas:     {stats['n_sesiones']}")
    print(f"  Registros analizados:    {stats['total_registros']}")
    print(f"  Duración total:          {stats['duracion_seg']} segundos")
    print(f"  Emoción predominante:    {stats['emocion_dominante']}")
    print(f"  Clima positivo:          {stats['categoria_pct'].get('Positiva', 0):.1f}%")
    print(f"  Clima neutro:            {stats['categoria_pct'].get('Neutra', 0):.1f}%")
    print(f"  Clima negativo:          {stats['categoria_pct'].get('Negativa', 0):.1f}%")
    print(f"  Estudiantes (promedio):  {stats['prom_estudiantes']}")

    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print(f"  Carpeta de salida: data/resultados/")
    print(f"  Imagen: {os.path.basename(salida_png)}")
    print(f"  CSV:    {os.path.basename(salida_csv)}")
    print("=" * 60)


if __name__ == "__main__":
    main()