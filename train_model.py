import os
import numpy as np
import cv2
import tensorflow as tf
from pycocotools.coco import COCO
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import seaborn as sns
import traceback


dataset_dir = 'pklot_dataset'
train_dir = os.path.join(dataset_dir, 'train')
valid_dir = os.path.join(dataset_dir, 'valid')
test_dir = os.path.join(dataset_dir, 'test')


train_annotations = os.path.join(train_dir, '_annotations.coco.json')
valid_annotations = os.path.join(valid_dir, '_annotations.coco.json')
test_annotations = os.path.join(test_dir, '_annotations.coco.json')


img_width, img_height = 150, 150
batch_size = 64  

def load_coco_annotations(annotations_file):
    coco = COCO(annotations_file)
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)
    annotations = coco.loadAnns(coco.getAnnIds())
    return images, annotations, coco



def data_generator(images, annotations, coco, directory, batch_size, img_width, img_height):
    while True:
        batch_images = []
        batch_labels = []
        for img in images:
            img_path = os.path.join(directory, img['file_name'])
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue  # Ignorar si la imagen no se puede cargar
                image = cv2.resize(image, (img_width, img_height))
                image = image.astype('float32') / 255.0

                # Obtener la etiqueta de la anotación
                annotation = [ann for ann in annotations if ann['image_id'] == img['id']]
                if annotation:
                    label = 1 if annotation[0]['category_id'] == 2 else 0  # Ajustar la categoría según tu necesidad
                    batch_images.append(image)
                    batch_labels.append(label)

                if len(batch_images) == batch_size:
                    yield np.array(batch_images), np.array(batch_labels)
                    batch_images = []
                    batch_labels = []
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                traceback.print_exc()
                continue

train_images, train_annotations, train_coco = load_coco_annotations(train_annotations)
valid_images, valid_annotations, valid_coco = load_coco_annotations(valid_annotations)
test_images, test_annotations, test_coco = load_coco_annotations(test_annotations)


# Combinar imágenes y anotaciones de validación y prueba
combined_images = valid_images + test_images
combined_annotations = valid_annotations + test_annotations

# Imprimir el número de imágenes y anotaciones en cada conjunto
print(f'Número de imágenes de entrenamiento: {len(train_images)}')
print(f'Número de anotaciones de entrenamiento: {len(train_annotations)}')
print(f'Número de imágenes de validación: {len(valid_images)}')
print(f'Número de anotaciones de validación: {len(valid_annotations)}')
print(f'Número de imágenes de prueba: {len(test_images)}')
print(f'Número de anotaciones de prueba: {len(test_annotations)}')
print(f'Número de imágenes combinadas: {len(combined_images)}')
print(f'Número de anotaciones combinadas: {len(combined_annotations)}')

train_generator = data_generator(train_images, train_annotations, train_coco, train_dir, batch_size, img_width, img_height)
valid_generator = data_generator(valid_images, valid_annotations, valid_coco, valid_dir, batch_size, img_width, img_height)
test_generator = data_generator(test_images, test_annotations, test_coco, test_dir, batch_size, img_width, img_height)
combined_generator = data_generator(combined_images, combined_annotations, valid_coco, valid_dir, batch_size, img_width, img_height)

# Cargar el modelo VGG-16 preentrenado en ImageNet, sin las capas superiores (fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Congelar las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Añadir nuestras propias capas FC para la clasificación binaria
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)  # Reducir el tamaño de las capas densas para acelerar
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Crear el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_generator,
          steps_per_epoch=len(train_images) // batch_size,
          validation_data=valid_generator,
          validation_steps=len(valid_images) // batch_size,
          epochs=5)  # Reducir el número de épocas para la prueba inicial

# Guardar el modelo
model.save('parking_lot_model.h5')

# Ajuste fino con imágenes combinadas
model.fit(combined_generator,
          steps_per_epoch=len(combined_images) // batch_size,
          epochs=5)

# Obtener las predicciones en el conjunto de prueba
y_true = []
y_pred = []

for images, labels in test_generator:
    predictions = model.predict(images)
    y_true.extend(labels)
    y_pred.extend(predictions > 0.5)  # Convertir probabilidades en etiquetas binarias
    if len(y_true) >= len(test_images):
        break

y_true = np.array(y_true[:len(test_images)])
y_pred = np.array(y_pred[:len(test_images)])

# Generar el reporte de clasificación
report = classification_report(y_true, y_pred, target_names=['Empty', 'Occupied'], digits=2)
print(report)

# Generar la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Empty', 'Occupied'], yticklabels=['Empty', 'Occupied'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


