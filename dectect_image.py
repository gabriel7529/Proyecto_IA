import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Cargar el modelo guardado
model = tf.keras.models.load_model('final_model.keras')

# Función para cargar y procesar una imagen
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Ruta de la imagen que quieres probar
test_image_path = 'uploads/parqueo.jpg'  # Cambia esto por la ruta de tu imagen

# Cargar y preprocesar la imagen
test_image = load_and_preprocess_image(test_image_path)

# Realizar la predicción
prediction = model.predict(test_image)
predicted_label = 'space-occupied' if prediction[0][0] > 0.5 else 'space-empty'

# Mostrar la imagen y la predicción
plt.imshow(load_img(test_image_path))
plt.title(f'Predicción: {predicted_label}')
plt.axis('off')
plt.show()

