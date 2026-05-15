import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Configuración de parámetros
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 25 # Puedes subirlo a 50 si tienes tiempo
TRAIN_DIR = 'data/raw/archive/Training/Training'
TEST_DIR = 'data/raw/archive/Testing/Testing'

# 2. Preprocesamiento de datos (Data Augmentation)
# Esto ayuda a que el modelo no se "aprenda de memoria" las fotos (overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 3. Definición de la Arquitectura CNN
model = models.Sequential([
    # Primera capa: Detecta bordes y formas simples
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Segunda capa: Formas más complejas
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Tercera capa
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Aplanar y capas densas para clasificar
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Evita el sobreajuste
    layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Entrenamiento
print("Iniciando entrenamiento...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

# 5. Guardar el modelo
model.save('models/emotion_model.h5')
print("Modelo guardado en models/emotion_model.h5")