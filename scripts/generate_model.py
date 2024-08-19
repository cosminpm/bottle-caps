import os
import uuid
from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from keras import Sequential
from keras.src.applications.resnet import ResNet50
from keras.src.layers import Dense, Flatten
from keras.src.saving import load_model

from app.services.identify.pinecone_container import PineconeContainer, image_to_vector
from app.shared.utils import read_img_from_path_with_mask, read_img_with_mask

PROJECT_PATH = os.getcwd()
load_dotenv()


def create_img_training(name: str, folder_create: str, path_all_images: str) -> None:
    """Create the training image for the model.

    Args:
    ----
        name: The name of the cap.
        folder_create: Where to create the img.
        path_all_images: The full path.

    """
    folder_name = os.path.splitext(name)[0]
    folder_result = os.path.join(folder_create, folder_name)

    if not os.path.exists(folder_result):
        os.makedirs(folder_result)
        path_img = str(Path(path_all_images) / name)
        img = read_img_from_path_with_mask(path_img)
        cv2.imwrite(str(Path(folder_result) / name), img)


def create_training_folder() -> None:
    """Create the training folder that's going to be used to train the model."""
    path_all_images = str(Path("database") / "caps-resized")
    folder_create = str(Path("database") / "training")

    names_images = os.listdir(path=path_all_images)
    for name in names_images:
        create_img_training(name=name, folder_create=folder_create, path_all_images=path_all_images)


def create_model() -> Sequential:
    """Create the Keras model that it's going to be used.

    Returns
    -------
        A Keras model to indentify bottle caps.

    """
    img_size = 224
    model = Sequential()
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
        pooling="max",
    )
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))  # Add fully connected layers

    model.compile("adam", loss=tf.losses.CategoricalCrossentropy(), metrics=["accuracy"])
    model.summary()
    return model


def transform_imag_to_pinecone_format(img: np.ndarray, model: keras.Sequential, metadata) -> dict:
    """Transform an image to pinecone format, so we can upload it into the vector database.

    Args:
    ----
        img: The image.
        model: The keras model.
        metadata: The medatada of the model.

    Returns:
    -------
        A dictionary with all the metadata information frpm pinecone.

    """
    img = read_img_with_mask(img)
    vector = image_to_vector(img=img, model=model)

    return {"id": str(uuid.uuid4()), "values": vector, "metadata": metadata}


def generate_vector_database(
    pinecone_container: PineconeContainer, model: keras.Sequential
) -> None:
    """Create the vector database for pinecone connection.

    Args:
    ----
        pinecone_container: The pinecone container.
        model: The keras model

    """
    root_dir = str(Path("database") / "training")
    folders = os.listdir(root_dir)
    for folder in folders:
        folder_path = str(Path(root_dir) / folder)
        for file in os.listdir(folder_path):
            path = str(Path(folder_path) / file)
            img = read_img_from_path_with_mask(path)
            vector = image_to_vector(img=img, model=model)
            cap_info = {"id": file, "values": vector}
            pinecone_container.upsert_to_pinecone(cap_info=cap_info)


def get_model() -> keras.Sequential:
    """Get the model.

    Returns
    -------
        The keras model.

    """
    path = str(Path(PROJECT_PATH) / "model.keras")
    return load_model(path)


def generate_model(pinecone_container: PineconeContainer) -> None:
    """Generate the model where we are going to save the bottle caps and the model used to identify.

    Args:
    ----
        pinecone_container: The pinecone container.


    """
    model = create_model()
    path_model: str = str(Path(PROJECT_PATH) / "model.keras")
    model.save(path_model)
    model = get_model()
    generate_vector_database(pinecone_container=pinecone_container, model=model)


def identify_cap(
    cap: np.ndarray,
    pinecone_con: PineconeContainer,
    model: keras.Sequential,
):
    """Identify a cap from the Pinecone database.

    Args:
    ----
        cap: The cap.
        pinecone_con: The Pinecone connection.
        model: The Keras model.

    Returns:
    -------
        The cap model with all the information.

    """
    img = read_img_with_mask(cap)
    vector = image_to_vector(img=img, model=model)
    result = pinecone_con.query_with_metadata(vector=vector)
    return [cap.to_dict() for cap in result]


if __name__ == "__main__":
    create_training_folder()
    pinecone_container = PineconeContainer()
    generate_model(pinecone_container=pinecone_container)
