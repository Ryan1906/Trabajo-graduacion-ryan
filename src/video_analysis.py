import cv2
import numpy as np
import mss
import pandas as pd
from tensorflow.keras.models import load_model
import time

# 1. Configuración del Modelo y Diccionario
model = load_model('models/emotion_model.h5')
emotion_dict = {0: "Enojado", 1: "Miedo", 2: "Feliz", 3: "Neutral", 4: "Triste", 5: "Sorpresa"}

# 2. Configuración de Captura de Pantalla
# Define el área de la pantalla a capturar (ajusta según tu ventana de Meet/Zoom)
monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
sct = mss.mss()

# Detector de rostros
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Selecciona la ventana de tu videollamada y presiona 'q' en la ventana de Python para salir.")

data_log = []
start_time = time.time()

while True:
    # Capturar pantalla (puedes configurar sct.monitors[1] para pantalla completa)
    monitor = sct.monitors[1] 
    screenshot = sct.grab(monitor)
    
    # Convertir a formato OpenCV
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la pantalla capturada
    faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

    current_emotions = []

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)
        label = emotion_dict[np.argmax(prediction)]
        current_emotions.append(label)

        # Dibujar feedback visual en la "ventana de monitoreo"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Lógica de Retroalimentación (Objetivo 2)
    if current_emotions:
        # Calcular la emoción predominante del grupo en este instante
        predominant = max(set(current_emotions), key=current_emotions.count)
        cv2.putText(frame, f"Estado Grupal: {predominant}", (50, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
        
        # Guardar para el reporte final
        data_log.append({
            "timestamp": time.time() - start_time,
            "emocion_grupal": predominant,
            "estudiantes_detectados": len(current_emotions)
        })

    # Redimensionar la vista previa para que no ocupe toda la pantalla del docente
    preview = cv2.resize(frame, (640, 360))
    cv2.imshow('Dashboard de Emociones (Docente)', preview)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guardar resultados de la sesión real
df = pd.DataFrame(data_log)
df.to_csv('data/sesion_tiempo_real.csv', index=False)
cv2.destroyAllWindows()