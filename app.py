import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Configuración del servidor Flask
app = Flask(__name__)

# Ruta del modelo guardado
model_path = 'parking_lot_model.h5'

# Cargar el modelo
model = load_model(model_path)

# Dimensiones de las imágenes
img_height, img_width = 224, 224

# Crear el directorio de subidas si no existe
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

def preprocess_image(img_path, target_size=(img_height, img_width)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalizar la imagen
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    img_file = request.files['file']
    
    if img_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    img_path = os.path.join(upload_folder, img_file.filename)
    img_file.save(img_path)

    img = preprocess_image(img_path)
    prediction = model.predict(img)
    
    os.remove(img_path)  # Eliminar la imagen después de la predicción
    
   
    result = int(prediction[0][0] > 0.5)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
