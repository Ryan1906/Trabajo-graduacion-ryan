# Prototipo de Análisis Emocional en Videollamadas Educativas

Este proyecto permite analizar emociones en tiempo real durante videollamadas educativas, mostrando la emoción grupal detectada en pantalla. Funciona en macOS, Windows y Linux.

## Requisitos
- Python 3.8+
- Dependencias: ver requirements.txt
- Modelo entrenado: `models/emotion_model.h5`

## Instalación
1. Clona el repositorio y entra al directorio:
	```sh
	git clone https://github.com/Ryan1906/Trabajo-graduacion-ryan.git
	cd Trabajo-graduacion-ryan
	```
2. Crea un entorno virtual y actívalo:
	```sh
	python3 -m venv venv
	source venv/bin/activate  # En Windows: venv\Scripts\activate
	```
3. Instala las dependencias:
	```sh
	pip install -r requirements.txt
	```
4. Verifica que el archivo `models/emotion_model.h5` existe.

## Uso de real_time_interface.py

1. Ejecuta la interfaz:
	```sh
	python src/real_time_interface.py
	```
2. Si usas macOS, otorga permiso de "Grabación de pantalla" a Python en:
	- Ajustes del Sistema → Privacidad y Seguridad → Grabación de pantalla

3. Haz clic en "INICIAR ANÁLISIS". El sistema detectará rostros en la videollamada activa y mostrará la emoción grupal en tiempo real.

4. Al detener el análisis, se guardará un archivo CSV con el registro de emociones detectadas en `data/sesion_tiempo_real.csv`.

## Notas
- No es necesario compartir pantalla ni grabar la videollamada, solo tener la ventana visible.
- Si tienes problemas con la cámara o permisos, revisa los mensajes en la interfaz.

## Autor
Ryan José Rodrigo Sigüenza Huertas

---

Trabajo de Graduación - Maestría en TIC
Universidad de San Carlos de Guatemala
Facultad de Ingeniería