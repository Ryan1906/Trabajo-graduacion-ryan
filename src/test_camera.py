import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Cargar el modelo que acabas de entrenar
model = load_model('models/emotion_model.h5')

# 2. Diccionario de emociones (Asegúrate de que el orden coincida con tus carpetas)
# El orden suele ser alfabético según las carpetas de Training
emotion_dict = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprise"}

# 3. Cargar el detector de rostros de OpenCV
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1) # Abrir cámara web

while True:
    ret, frame = cap.read()
    
    # SI EL FRAME ESTÁ VACÍO, SALTAR AL SIGUIENTE INTENTO
    if not ret or frame is None:
        continue 
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros
    faces = face_classifier.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Dibujar rectángulo en el rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Recortar y preparar la cara para el modelo
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0 # Normalizar
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1) # Ajustar forma a (1, 48, 48, 1)

            # Predicción
            prediction = model.predict(roi)
            label = emotion_dict[np.argmax(prediction)]
            
            # Escribir la emoción en pantalla
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Detector de Emociones - Tesis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Presiona 'q' para salir
        break

cap.release()
cv2.destroyAllWindows()