import json
import math
import os
import shutil
from typing import List, Dict, Any

import pinecone
import numpy as np
from PIL import Image
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from ScriptsMain.DetectCaps import detect_caps
from ScriptsMain.utilsFun import read_img_from_path

PROJECT_PATH = os.getcwd()


def create_training_folder():
    path_all_images = os.path.join(PROJECT_PATH, r'database\caps-resized')
    folder_create = os.path.join(PROJECT_PATH, 'training')

    names_images = os.listdir(path=path_all_images)
    for name in names_images:
        folder_name = name.split('.')[0]
        folder_result = os.path.join(folder_create, folder_name)
        os.mkdir(folder_result)
        shutil.copy(os.path.join(path_all_images, name), os.path.join(folder_result))


def create_model():
    img_size = 224
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3), pooling='max')
    model.compile()
    save_model(model=model, path=os.path.join(PROJECT_PATH, 'model'))


def generate_vector_database(index: pinecone.Index, model):
    root_dir = os.path.join(PROJECT_PATH, 'training')
    batch_size = 64
    img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    datagen = img_gen.flow_from_directory(root_dir,
                                          target_size=(224, 224),
                                          batch_size=batch_size,
                                          class_mode=None,
                                          shuffle=False)
    num_images = len(datagen.filenames)
    num_epochs = int(math.ceil(num_images / batch_size))

    feature_list = model.predict_generator(datagen, num_epochs)

    json_path = os.path.join(PROJECT_PATH, 'vector_database_pinecone.json')

    json_object = {
        "vectors": [],
        "namespace": "bottle_caps"
    }

    for i in range(0, len(feature_list)):
        cap_info = {
            'id': datagen.filenames[i],
            'values': feature_list[i].tolist()
        }
        upsert_to_pinecone(index=index, vector=cap_info)
        json_object['vectors'].append(cap_info)

    with open(json_path, 'w') as json_file:
        json.dump(json_object, json_file)


def upsert_to_pinecone(index: pinecone.Index, vector):
    index.upsert(
        vectors=[
            vector
        ],
        namespace='bottle_caps'
    )


def save_model(model, path):
    model.save(path)


def get_model():
    path = os.path.join(PROJECT_PATH, 'model')
    return load_model(path)


def generate_all(index: pinecone.Index):
    create_model()
    model = get_model()
    generate_vector_database(index=index, model=model)


def path_to_vector(model, img_path: str):
    img_path = os.path.join(PROJECT_PATH, img_path)
    img = Image.open(img_path)
    return image_to_vector(img, model)


def image_to_vector(img, model):
    resized_img = np.resize(img, (224, 224, 3))  # Resize the image to (224, 224)
    preprocessed_img = preprocess_input(resized_img[np.newaxis, ...])  # Preprocess the resized image
    query_feature = model.predict(preprocessed_img)
    return query_feature[0].tolist()


def query_path_pinecone(model, img_path: str, index: pinecone.Index):
    vector = path_to_vector(model, img_path)
    return index.query(vector=[vector], top_k=15, namespace="bottle_caps")


def query_img_pinecone(model, img, index: pinecone.Index):
    vector = image_to_vector(model=model, img=img)
    return index.query(vector=[vector], top_k=15, namespace="bottle_caps")


def init_index_pinecone() -> pinecone.Index:
    pinecone.init(api_key=os.environ["API_KEY"], environment=os.environ["ENV"])
    return pinecone.Index(index_name='bottle-caps')


def main():
    index = init_index_pinecone()
    model = get_model()
    # generate_all(index=index)

    path = os.path.join(PROJECT_PATH, r'database/my-caps-images/8-fresh.jpg')
    img = read_img_from_path(path)
    cropped_caps = detect_caps(img)

    # for img, rct in cropped_caps:
    #     result = query_img_pinecone(model=model, img=img, index=index)
    #     print(result)
    result = query_path_pinecone(model=model, img_path=path, index=index)
    print(result)

if __name__ == '__main__':
    # generate_all()
    # use_model()
    main()
