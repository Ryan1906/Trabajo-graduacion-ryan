"""
Prototipo de Análisis Emocional en Videollamadas Educativas
============================================================
Trabajo de Graduación - Maestría en TIC
Universidad de San Carlos de Guatemala
Facultad de Ingeniería

Autor: Ryan José Rodrigo Sigüenza Huertas

Compatible con macOS, Windows y Linux.
En macOS requiere otorgar permiso de "Grabación de pantalla"
a la aplicación que ejecuta Python en:
Ajustes del Sistema → Privacidad y Seguridad → Grabación de pantalla.

Los CSV de cada sesión se guardan en: data/sesiones/
"""

import os
import re
import sys
import time
import platform
import subprocess
import tempfile
import threading
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, simpledialog

from tensorflow.keras.models import load_model

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


# ====================================================================
# CONFIGURACIÓN Y ESTRUCTURA DE CARPETAS
# ====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.h5')

# Estructura de datos ordenada
DATA_DIR = os.path.join(BASE_DIR, 'data')
SESIONES_DIR = os.path.join(DATA_DIR, 'sesiones')      # CSV de cada grabación
RESULTADOS_DIR = os.path.join(DATA_DIR, 'resultados')  # gráficas y resúmenes

# Crear las carpetas si no existen
for carpeta in (DATA_DIR, SESIONES_DIR, RESULTADOS_DIR):
    os.makedirs(carpeta, exist_ok=True)

EMOTION_DICT = {
    0: "Enojado", 1: "Miedo", 2: "Feliz",
    3: "Neutral", 4: "Triste", 5: "Sorpresa"
}

# Parámetros de detección. Permisivos para rostros pequeños en
# pantallas compartidas tipo Zoom/Meet.
DETECT_MIN_SIZE = (30, 30)
DETECT_SCALE_FACTOR = 1.1
DETECT_MIN_NEIGHBORS = 4

IS_MAC = platform.system() == "Darwin"


def slugify(texto):
    """Convierte un nombre de sesión en un nombre de archivo seguro."""
    texto = texto.strip().lower()
    texto = re.sub(r'[^\w\s-]', '', texto)
    texto = re.sub(r'[\s]+', '_', texto)
    return texto or "sesion"


# ====================================================================
# CAPTURADOR DE PANTALLA MULTIPLATAFORMA
# ====================================================================
class ScreenCapturer:
    def __init__(self):
        self.backend = None
        self.sct = None
        self.monitor = None
        self._init_backend()

    def _init_backend(self):
        if MSS_AVAILABLE:
            try:
                self.sct = mss.mss()
                self.monitor = self.sct.monitors[1]
                test = np.array(self.sct.grab(self.monitor))
                if test.size == 0 or test.mean() < 1:
                    raise RuntimeError("mss devolvió un frame vacío/negro")
                self.backend = "mss"
                print(f"[ScreenCapturer] Backend: mss  |  Monitor: {self.monitor}")
                return
            except Exception as e:
                print(f"[ScreenCapturer] mss falló: {e}. Intentando fallback…")
                self.sct = None

        if IS_MAC:
            self.backend = "screencapture"
            print("[ScreenCapturer] Backend: screencapture (nativo de macOS)")
            return

        try:
            import pyautogui  # noqa: F401
            self.backend = "pyautogui"
            print("[ScreenCapturer] Backend: pyautogui")
            return
        except ImportError:
            pass

        raise RuntimeError("No se pudo inicializar la captura de pantalla.")

    def grab(self):
        if self.backend == "mss":
            img = np.array(self.sct.grab(self.monitor))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if self.backend == "screencapture":
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            try:
                subprocess.run(
                    ["screencapture", "-x", "-t", "jpg", tmp.name],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                frame = cv2.imread(tmp.name)
                if frame is None:
                    raise RuntimeError("screencapture devolvió imagen vacía.")
                return frame
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        if self.backend == "pyautogui":
            import pyautogui
            img = np.array(pyautogui.screenshot())
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        raise RuntimeError("Backend de captura no configurado")

    def close(self):
        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass


# ====================================================================
# APLICACIÓN PRINCIPAL
# ====================================================================
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prototipo de Análisis Emocional - Tesis USAC")
        self.root.geometry("780x760")
        self.root.configure(bg="#1e1e2e")

        self.analyzing = False
        self.data_log = []
        self.start_time = None
        self.capturer = None
        self.session_name = None
        self.session_started_at = None

        try:
            print(f"[App] Cargando modelo desde: {MODEL_PATH}")
            self.model = load_model(MODEL_PATH, compile=False)

            haar_dir = cv2.data.haarcascades
            self.face_frontal = cv2.CascadeClassifier(
                haar_dir + 'haarcascade_frontalface_default.xml'
            )
            self.face_alt = cv2.CascadeClassifier(
                haar_dir + 'haarcascade_frontalface_alt2.xml'
            )
            self.face_profile = cv2.CascadeClassifier(
                haar_dir + 'haarcascade_profileface.xml'
            )
            if self.face_frontal.empty() or self.face_profile.empty():
                raise RuntimeError("No se cargaron los clasificadores Haar")
        except Exception as e:
            messagebox.showerror(
                "Error al cargar modelos",
                f"No se pudo cargar el modelo o los clasificadores:\n\n{e}"
            )
            sys.exit(1)

        self._build_ui()

        if IS_MAC:
            self.root.after(500, self._show_mac_permissions_tip)

    # ----------------------------- UI ---------------------------------
    def _build_ui(self):
        tk.Label(
            self.root,
            text="Análisis Emocional en Tiempo Real",
            font=("Helvetica", 16, "bold"),
            bg="#1e1e2e", fg="#cdd6f4"
        ).pack(pady=(15, 5))

        tk.Label(
            self.root,
            text="Captura la videollamada activa y muestra la emoción grupal",
            font=("Helvetica", 10),
            bg="#1e1e2e", fg="#a6adc8"
        ).pack(pady=(0, 10))

        self.btn = tk.Button(
            self.root,
            text="▶  INICIAR ANÁLISIS",
            command=self.toggle,
            bg="#2ecc71", fg="black",
            font=("Helvetica", 12, "bold"),
            height=2, width=25,
            relief="flat", cursor="hand2"
        )
        self.btn.pack(pady=10)

        self.label_sesion = tk.Label(
            self.root,
            text="Sesión: (ninguna)",
            font=("Helvetica", 11, "bold"),
            bg="#1e1e2e", fg="#94e2d5"
        )
        self.label_sesion.pack(pady=(0, 4))

        self.canvas = tk.Label(self.root, bg="black", width=640, height=360)
        self.canvas.pack(pady=10)

        self.label_emocion = tk.Label(
            self.root,
            text="Estado: Esperando…",
            font=("Helvetica", 16, "bold"),
            bg="#1e1e2e", fg="#f9e2af"
        )
        self.label_emocion.pack(pady=8)

        self.label_info = tk.Label(
            self.root,
            text="Rostros detectados: 0   |   FPS: 0.0   |   Tiempo: 00:00",
            font=("Helvetica", 10),
            bg="#1e1e2e", fg="#a6adc8"
        )
        self.label_info.pack(pady=2)

        self.label_status = tk.Label(
            self.root,
            text="",
            font=("Helvetica", 9, "italic"),
            bg="#1e1e2e", fg="#89b4fa",
            wraplength=740, justify="center"
        )
        self.label_status.pack(pady=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _show_mac_permissions_tip(self):
        msg = (
            "macOS detectado. Si la captura sale negra, otorga permiso "
            "de Grabación de Pantalla en Ajustes del Sistema → "
            "Privacidad y Seguridad → Grabación de Pantalla."
        )
        self.label_status.config(text=msg)

    # --------------------------- Control ------------------------------
    def toggle(self):
        if not self.analyzing:
            self._start()
        else:
            self._stop()

    def _start(self):
        nombre = simpledialog.askstring(
            "Nueva sesión",
            "Nombre o identificador de esta sesión\n"
            "(ej. 'Clase Matematica 5A', 'Grupo Control', 'Prueba 1'):",
            parent=self.root
        )
        if nombre is None:
            return
        if not nombre.strip():
            nombre = "sesion_sin_nombre"

        self.session_name = nombre.strip()
        self.session_started_at = datetime.now()

        try:
            self.capturer = ScreenCapturer()
        except Exception as e:
            messagebox.showerror(
                "Error de captura de pantalla",
                f"No se pudo iniciar la captura:\n\n{e}"
            )
            return

        self.analyzing = True
        self.start_time = time.time()
        self.data_log = []
        self.btn.config(text="■  DETENER ANÁLISIS", bg="#e74c3c")
        self.label_sesion.config(text=f"Sesión: {self.session_name}")
        self.label_status.config(
            text=f"Capturando con backend: {self.capturer.backend}  ·  "
                 f"Inicio: {self.session_started_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        threading.Thread(target=self._loop_analisis, daemon=True).start()

    def _stop(self):
        self.analyzing = False
        self.btn.config(text="▶  INICIAR ANÁLISIS", bg="#2ecc71")
        if self.capturer is not None:
            self.capturer.close()
            self.capturer = None
        self._save_data()

    def _on_close(self):
        self.analyzing = False
        if self.capturer is not None:
            self.capturer.close()
        self._save_data()
        self.root.destroy()

    # --------------------- Detección multi-cascade --------------------
    def _detect_faces(self, gray):
        """Combina 3 cascades (frontal default, alt2 y perfil) + NMS."""
        all_faces = []

        for cascade in (self.face_frontal, self.face_alt, self.face_profile):
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=DETECT_SCALE_FACTOR,
                minNeighbors=DETECT_MIN_NEIGHBORS,
                minSize=DETECT_MIN_SIZE,
            )
            for f in faces:
                all_faces.append(tuple(f))

        gray_flipped = cv2.flip(gray, 1)
        w = gray.shape[1]
        faces_flipped = self.face_profile.detectMultiScale(
            gray_flipped,
            scaleFactor=DETECT_SCALE_FACTOR,
            minNeighbors=DETECT_MIN_NEIGHBORS,
            minSize=DETECT_MIN_SIZE,
        )
        for (x, y, fw, fh) in faces_flipped:
            all_faces.append((w - x - fw, y, fw, fh))

        return self._non_max_suppression(all_faces, overlap_threshold=0.3)

    @staticmethod
    def _non_max_suppression(boxes, overlap_threshold=0.3):
        if not boxes:
            return []
        b = np.array(boxes, dtype=float)
        x1, y1 = b[:, 0], b[:, 1]
        x2, y2 = x1 + b[:, 2], y1 + b[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou < overlap_threshold]

        return [tuple(map(int, b[i])) for i in keep]

    # --------------------------- Loop ---------------------------------
    def _loop_analisis(self):
        frame_count = 0
        fps_t0 = time.time()
        fps_value = 0.0

        while self.analyzing:
            try:
                frame = self.capturer.grab()
            except Exception as e:
                self.root.after(0, lambda err=e: self.label_status.config(
                    text=f"Error de captura: {err}"
                ))
                time.sleep(0.5)
                continue

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_full = cv2.equalizeHist(gray_full)

            faces = self._detect_faces(gray_full)

            current_emotions = []
            for (x, y, fw, fh) in faces:
                if fw < 20 or fh < 20:
                    continue
                roi_gray = gray_full[y:y + fh, x:x + fw]
                if roi_gray.size == 0:
                    continue

                roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_resized.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                prediction = self.model.predict(roi, verbose=0)
                label = EMOTION_DICT[int(np.argmax(prediction))]
                current_emotions.append(label)

                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 3)
                cv2.putText(
                    frame, label, (x, max(y - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )

            elapsed = time.time() - self.start_time

            if current_emotions:
                predominant = max(set(current_emotions), key=current_emotions.count)
                conteo = {emo: current_emotions.count(emo) for emo in set(current_emotions)}

                self.data_log.append({
                    "sesion": self.session_name,
                    "fecha_inicio": self.session_started_at.strftime('%Y-%m-%d %H:%M:%S'),
                    "tiempo_seg": round(elapsed, 2),
                    "tiempo_mmss": self._fmt_mmss(elapsed),
                    "emocion_grupal": predominant,
                    "estudiantes_detectados": len(current_emotions),
                    "n_enojado": conteo.get("Enojado", 0),
                    "n_miedo": conteo.get("Miedo", 0),
                    "n_feliz": conteo.get("Feliz", 0),
                    "n_neutral": conteo.get("Neutral", 0),
                    "n_triste": conteo.get("Triste", 0),
                    "n_sorpresa": conteo.get("Sorpresa", 0),
                })
            else:
                predominant = "Sin rostros"

            frame_count += 1
            if frame_count >= 5:
                now = time.time()
                fps_value = frame_count / (now - fps_t0)
                fps_t0 = now
                frame_count = 0

            h, w = frame.shape[:2]
            preview_w = 640
            preview_h = int(h * (preview_w / w))
            preview = cv2.resize(frame, (preview_w, preview_h))

            img_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.root.after(
                0,
                self._update_ui,
                img_tk,
                predominant,
                len(current_emotions),
                fps_value,
                elapsed,
            )

            time.sleep(0.05)

    @staticmethod
    def _fmt_mmss(segundos):
        """Convierte segundos a formato mm:ss."""
        m = int(segundos // 60)
        s = int(segundos % 60)
        return f"{m:02d}:{s:02d}"

    def _update_ui(self, img_tk, emocion, n_caras, fps, elapsed):
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk
        self.label_emocion.config(text=f"Emoción grupal: {emocion}")
        self.label_info.config(
            text=f"Rostros detectados: {n_caras}   |   "
                 f"FPS: {fps:.1f}   |   "
                 f"Tiempo: {self._fmt_mmss(elapsed)}"
        )

    # ------------------------- Persistencia ---------------------------
    def _save_data(self):
        if not self.data_log:
            self.label_status.config(
                text="No hay datos que guardar (no se detectaron rostros)."
            )
            return

        # Guardar en data/sesiones/
        slug = slugify(self.session_name)
        timestamp = self.session_started_at.strftime('%Y%m%d_%H%M%S')
        filename = f"sesion_{slug}_{timestamp}.csv"
        out = os.path.join(SESIONES_DIR, filename)

        pd.DataFrame(self.data_log).to_csv(out, index=False)
        print(f"[App] Datos guardados en: {out} ({len(self.data_log)} registros)")
        self.label_status.config(
            text=f"Sesión guardada: sesiones/{filename}  ·  "
                 f"{len(self.data_log)} registros"
        )
        messagebox.showinfo(
            "Sesión guardada",
            f"Los datos de la sesión '{self.session_name}' se guardaron en:\n\n"
            f"data/sesiones/{filename}\n\n"
            f"Registros capturados: {len(self.data_log)}\n\n"
            f"Para generar las gráficas ejecuta:\n"
            f"python src/generate_results.py"
        )


# ====================================================================
# ENTRY POINT
# ====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()