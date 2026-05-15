import cv2
import numpy as np
import mss
import pandas as pd
from tensorflow.keras.models import load_model
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk

# --- CONFIGURACIÓN DE MODELOS ---
MODEL_PATH = 'models/emotion_model.h5'
emotion_dict = {0: "Enojado", 1: "Miedo", 2: "Feliz", 3: "Neutral", 4: "Triste", 5: "Sorpresa"}
model = load_model(MODEL_PATH)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prototipo de Análisis Emocional - Tesis")
        self.root.geometry("700x600")
        
        self.analyzing = False
        self.data_log = []

        # --- DISEÑO DE LA INTERFAZ ---
        self.btn = tk.Button(root, text="INICIAR ANÁLISIS", command=self.toggle, 
                             bg="#2ecc71", fg="black", font=("Arial", 12, "bold"), height=2)
        self.btn.pack(pady=10)

        # Este es el lienzo donde se verá la pantalla capturada
        self.canvas = tk.Label(root, bg="black")
        self.canvas.pack(pady=10, expand=True)

        self.label_emocion = tk.Label(root, text="Estado: Esperando...", font=("Arial", 18, "bold"))
        self.label_emocion.pack(pady=10)

    def toggle(self):
        if not self.analyzing:
            self.analyzing = True
            self.btn.config(text="DETENER ANÁLISIS", bg="#e74c3c")
            threading.Thread(target=self.loop_analisis, daemon=True).start()
        else:
            self.analyzing = False
            self.btn.config(text="INICIAR ANÁLISIS", bg="#2ecc71")
            self.save_data()

    def loop_analisis(self):
        with mss.mss() as sct:
            # Captura el monitor principal
            monitor = sct.monitors[1]
            
            while self.analyzing:
                # 1. Capturar pantalla
                img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Redimensionar para que quepa en la interfaz (importante)
                frame_small = cv2.resize(frame, (640, 360))
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                
                # 2. Detectar Rostros
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                emocion_actual = "Neutral"

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi = roi_gray.astype('float') / 255.0
                    roi = np.expand_dims(roi, axis=0)
                    roi = np.expand_dims(roi, axis=-1)

                    prediction = model.predict(roi, verbose=0)
                    emocion_actual = emotion_dict[np.argmax(prediction)]

                    # Dibujar en el preview
                    cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_small, emocion_actual, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 3. Actualizar la Interfaz (Imagen y Texto)
                # Convertir frame para Tkinter
                img_p = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                img_p = Image.fromarray(img_p)
                img_tk = ImageTk.PhotoImage(image=img_p)

                # Usamos after para actualizar la UI desde el hilo secundario
                self.root.after(0, self.update_ui, img_tk, emocion_actual)
                
                # Guardar log
                if faces is not None:
                    self.data_log.append({"tiempo": time.time(), "emocion": emocion_actual})

                time.sleep(0.1) # Control de velocidad

    def update_ui(self, img_tk, texto):
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk # Mantener referencia
        self.label_emocion.config(text=f"Emoción Grupal: {texto}")

    def save_data(self):
        if self.data_log:
            pd.DataFrame(self.data_log).to_csv('data/sesion_maestria.csv', index=False)
            print("Datos guardados para la tesis.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()