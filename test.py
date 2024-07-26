import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings



# Directorio del dataset y de anotaciones
dataset_dir = 'pklot_dataset'
test_dir = os.path.join(dataset_dir, 'test')

# Función para cargar anotaciones y agregar la ruta completa del archivo
def load_annotations(dir_path):
    annotations_path = os.path.join(dir_path, '_annotations.csv')
    annotations_df = pd.read_csv(annotations_path, header=None)
    annotations_df.columns = ['file_name', 'x_min', 'y_min', 'x_max', 'y_max', 'label']
    annotations_df['file_path'] = annotations_df.apply(lambda row: os.path.join(dir_path, row['file_name']), axis=1)
    return annotations_df

# Cargar las anotaciones de la carpeta de prueba
test_annotations = load_annotations(test_dir)

# Seleccionar un subconjunto aleatorio de datos para la prueba
test_sample_size = 1200
test_annotations = test_annotations.sample(n=test_sample_size, random_state=42)

# Crear una función para cargar y procesar las imágenes y anotaciones
def load_image(img_path, label):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

# Generar los datos de prueba
def data_generator(dataframe, batch_size, img_size):
    while True:
        for start in range(0, len(dataframe), batch_size):
            end = min(start + batch_size, len(dataframe))
            batch_data = dataframe[start:end]
            images = []
            labels = []
            for idx, row in batch_data.iterrows():
                img_path = row['file_path']
                label = 1 if row['label'] == 'space-occupied' else 0
                img, label = load_image(img_path, label)
                images.append(img)
                labels.append(label)
            yield np.array(images), np.array(labels)

# Crear generador de datos de prueba
batch_size = 32
test_gen = data_generator(test_annotations, batch_size, (224, 224))

# Calcular pasos por época
test_steps_per_epoch = len(test_annotations) // batch_size

# Cargar el modelo guardado
model = tf.keras.models.load_model('final_model.keras')

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_gen, steps=test_steps_per_epoch)
print(f'Test accuracy: {test_acc:.4f}')

# Predicciones en el conjunto de prueba
test_predictions = model.predict(test_gen, steps=test_steps_per_epoch)
test_predictions = np.round(test_predictions).astype(int)

# Generar etiquetas verdaderas del conjunto de prueba
test_labels = []
for start in range(0, len(test_annotations), batch_size):
    end = min(start + batch_size, len(test_annotations))
    batch_data = test_annotations[start:end]
    for idx, row in batch_data.iterrows():
        label = 1 if row['label'] == 'space-occupied' else 0
        test_labels.append(label)

# Asegurarse de que test_labels y test_predictions tengan la misma longitud
test_labels = np.array(test_labels[:len(test_predictions)])

# Generar matriz de confusión y reporte de clasificación
conf_matrix = confusion_matrix(test_labels, test_predictions)
class_report = classification_report(test_labels, test_predictions, target_names=['Empty', 'Occupied'])

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualizar matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Empty', 'Occupied'], yticklabels=['Empty', 'Occupied'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
