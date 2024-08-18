import os

import uuid
import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.applications.resnet import ResNet50
from keras.src.layers import Flatten, Dense
from keras.src.saving import load_model

from app.services.identify.manager import PineconeContainer, image_to_vector
from app.shared.utils import read_img_from_path_with_mask, read_img_with_mask

# PROJECT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PROJECT_PATH = os.getcwd()


def create_img_training(name: str, folder_create: str, path_all_images: str):
    folder_name = os.path.splitext(name)[0]
    folder_result = os.path.join(folder_create, folder_name)

    if not os.path.exists(folder_result):
        os.makedirs(folder_result)
        path_img = os.path.join(path_all_images, name)
        img = read_img_from_path_with_mask(path_img)
        cv2.imwrite(os.path.join(folder_result, name), img)


def create_training_folder():
    path_all_images = os.path.join(PROJECT_PATH, 'database', 'caps-resized')
    folder_create = os.path.join(PROJECT_PATH, 'training')

    names_images = os.listdir(path=path_all_images)
    for name in names_images:
        create_img_training(name=name, folder_create=folder_create, path_all_images=path_all_images)


def create_model():
    img_size = 224
    model = Sequential()
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3), pooling='max')
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))  # Add fully connected layers

    model.compile('adam', loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()
    return model


def transform_imag_to_pinecone_format(img: np.ndarray, model: keras.Sequential, metadata):
    img = read_img_with_mask(img)
    vector = image_to_vector(img=img, model=model)

    cap_info = {
        'id': str(uuid.uuid4()),
        'values': vector,
        'metadata': metadata
    }

    return cap_info


def generate_vector_database(pinecone_container, model: keras.Sequential):
    root_dir = os.path.join(PROJECT_PATH, 'training')
    folders = os.listdir(root_dir)
    for folder in folders:
        folder = os.path.join(root_dir, folder)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = read_img_from_path_with_mask(path)
            vector = image_to_vector(img=img, model=model)
            cap_info = {
                'id': file,
                'values': vector
            }
            pinecone_container.upsert_to_pinecone(cap_info=cap_info)


def save_model(model, path):
    model.save(path)


def get_model() -> keras.Sequential:
    path = os.path.join(PROJECT_PATH, 'model')
    return load_model(path)


def generate_all(pinecone_container: PineconeContainer):
    model = create_model()
    path_model = os.path.join(PROJECT_PATH, 'model')
    save_model(model=model, path=path_model)
    model = get_model()
    generate_vector_database(pinecone_container=pinecone_container, model=model)


def identify_cap(cap: np.ndarray, pinecone_con: PineconeContainer, model: keras.Sequential, user_id: str):
    img = read_img_with_mask(cap)
    vector = image_to_vector(img=img, model=model)
    metadata = {'user_id': {"$eq": user_id}}
    result = pinecone_con.query_with_metadata(vector=vector, metadata=metadata)
    return [cap.to_dict() for cap in result]


if __name__ == '__main__':
    create_training_folder()
    pinecone_container = PineconeContainer()
    generate_all(pinecone_container=pinecone_container)
