import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Directorios del dataset
dataset_dir = 'pklot_dataset'
train_dir = os.path.join(dataset_dir, 'train')
valid_dir = os.path.join(dataset_dir, 'valid')
test_dir = os.path.join(dataset_dir, 'test')

def load_annotations(dir_path):
    annotations_path = os.path.join(dir_path, '_annotations.csv')
    annotations_df = pd.read_csv(annotations_path, header=None)
    annotations_df.columns = ['file_name', 'x_min', 'y_min', 'x_max', 'y_max', 'label']
    annotations_df['file_path'] = annotations_df.apply(lambda row: os.path.join(dir_path, row['file_name']), axis=1)
    return annotations_df

train_annotations = load_annotations(train_dir)
valid_annotations = load_annotations(valid_dir)
test_annotations = load_annotations(test_dir)

def select_random_subset(df, sample_size):
    return df.sample(n=sample_size, random_state=42)

train_sample_size = 50000
valid_sample_size = 5000
test_sample_size = 1200

train_annotations = select_random_subset(train_annotations, train_sample_size)
valid_annotations = select_random_subset(valid_annotations, valid_sample_size)
test_annotations = select_random_subset(test_annotations, test_sample_size)

def load_image(img_path, label):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

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

batch_size = 32
train_gen = data_generator(train_annotations, batch_size, (224, 224))
valid_gen = data_generator(valid_annotations, batch_size, (224, 224))
test_gen = data_generator(test_annotations, batch_size, (224, 224))

train_steps_per_epoch = len(train_annotations) // batch_size
valid_steps_per_epoch = len(valid_annotations) // batch_size
test_steps_per_epoch = len(test_annotations) // batch_size

# Definición del modelo
HEIGHT, WIDTH = 224, 224
image_input = Input(shape=(HEIGHT, WIDTH, 3))

base_model = ResNet50(input_tensor=image_input, include_top=False, weights='imagenet')
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)  # Cambiado a 1 salida para clasificación binaria

model = Model(inputs=base_model.input, outputs=predictions)

# Congelar todas las capas de la base de ResNet50 excepto las últimas capas densas
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('finals_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    epochs=50,
    validation_data=valid_gen,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=valid_steps_per_epoch,
    callbacks=[checkpoint, early_stopping]
)

model.save('finals_model.keras')

