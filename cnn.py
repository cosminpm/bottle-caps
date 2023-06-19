import json
import math
import os
import shutil

import pinecone
import numpy as np
from PIL import Image
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

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


def generate_vector_database(model):
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
        json_object['vectors'].append(cap_info)

    with open(json_path, 'w') as json_file:
        json.dump(json_object, json_file)


def save_model(model, path):
    model.save(path)


def get_model():
    path = os.path.join(PROJECT_PATH, 'model')
    return load_model(path)


def generate_all():
    create_model()
    model = get_model()
    generate_vector_database(model=model)


def use_model():
    path = os.path.join(PROJECT_PATH, 'model')
    img_path = os.path.join(PROJECT_PATH, '9.jpg')

    model = load_model(path)
    img = Image.open(img_path)
    resized_img = img.resize((224, 224))  # Resize the image to (224, 224)
    resized_img = np.array(resized_img)
    preprocessed_img = preprocess_input(resized_img[np.newaxis, ...])  # Preprocess the resized image
    query_feature = model.predict(preprocessed_img)
    query_feature = query_feature[0]

    json_path = os.path.join(PROJECT_PATH, 'query_feature_value.json')
    with open(json_path, 'w') as json_file:
        json.dump(query_feature.tolist(), json_file)


def main():
    pinecone.init(api_key=os.environ["API_KEY"], environment=os.environ["ENVIRONMENT"])
    r = pinecone.list_indexes()
    print(r)

if __name__ == '__main__':
    # generate_all()
    # use_model()
    main()