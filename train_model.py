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


