import json
import math
import pickle
import shutil
import os
import numpy as np
from PIL import Image

from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model


def create_training_folder():
    path_all_images = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\database\caps-resized'
    folder_create = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\training'

    names_images = os.listdir(path=path_all_images)
    for name in names_images:
        folder_name = name.split('.')[0]
        folder_result = os.path.join(folder_create, folder_name)
        os.mkdir(folder_result)
        shutil.copy(os.path.join(path_all_images, name), os.path.join(folder_result))


def save_model(model, path):
    model.save(path)


def create_model():
    img_size = 224
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3), pooling='max')
    model.compile()
    save_model(model=model, path=r'C:\Users\manolito\Repositories\GitHub\BottleCaps\model')


def generate_vector_database(model):
    root_dir = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\training'
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

    # Apply PCA to reduce the dimensionality of each feature vector
    pca = PCA(n_components=64)
    reduced_feature_list = pca.fit_transform(feature_list)


    json_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\vector_database_pinecone.json'

    json_object = {
        "vectors": [],
        "namespace": "bottle_caps"
    }

    for i in range(0, len(feature_list)):
        cap_info = {
            'id': datagen.filenames[i],
            'values': reduced_feature_list[i].tolist()
        }
        json_object['vectors'].append(cap_info)

    print(json_object)
    with open(json_path, 'w') as json_file:
        json.dump(json_object, json_file)

    save_datagen_files(datagen.filenames)
    save_feature_list(reduced_feature_list)

def save_datagen_files(filenames: list[str]):
    json_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\datagen_filenames.json'
    with open(json_path, 'w') as json_file:
        json.dump(filenames, json_file)


def save_neighbours(feature_list):
    pkl_file = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\neighbors.pkl'

    neighbors = NearestNeighbors(algorithm='ball_tree', metric='euclidean')
    neighbors.fit(feature_list)
    with open(pkl_file, 'wb') as file:
        pickle.dump(neighbors, file)


def open_datagen_files():
    json_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\datagen_filenames.json'
    with open(json_path, 'r') as json_file:
        feature_list = np.array(json.load(json_file))
        return feature_list


def save_feature_list(feature_list):
    json_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\vector_database.json'
    with open(json_path, 'w') as json_file:
        json.dump(feature_list.tolist(), json_file)


def open_feature_list():
    json_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\vector_database.json'
    with open(json_path, 'r') as json_file:
        feature_list = np.array(json.load(json_file))
        return feature_list


def load_neighbors():
    # Load the NearestNeighbors object from a file
    neigbors_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\\neighbors.pkl'
    with open(neigbors_path, 'rb') as file:
        neighbors = pickle.load(file)
        return neighbors


def get_model():
    path = r"C:\Users\manolito\Repositories\GitHub\BottleCaps\model"
    return load_model(path)


def generate_all():
    create_model()
    model = get_model()
    generate_vector_database(model=model)
    feature_list = open_feature_list()
    save_neighbours(feature_list=feature_list)

from sklearn.decomposition import PCA

def use_model():
    path = r"C:\Users\manolito\Repositories\GitHub\BottleCaps\model"
    img_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\9.jpg'
    max_neighbours = 10

    model = load_model(path)
    img = Image.open(img_path)
    resized_img = img.resize((224, 224))  # Resize the image to (224, 224)
    resized_img = np.array(resized_img)
    preprocessed_img = preprocess_input(resized_img[np.newaxis, ...])  # Preprocess the resized image
    query_feature = model.predict(preprocessed_img)
    query_feature = query_feature.flatten()

    # Reshape the query_feature array to have 1 sample and multiple features
    query_feature = query_feature.reshape(1, -1)

    # Apply PCA to reduce dimensionality to length 64
    pca = PCA(n_components=min(64, query_feature.shape[1]))
    query_feature = pca.fit_transform(query_feature)

    print(query_feature)

    #
    # feature_list = open_feature_list()
    # neighbors = load_neighbors()
    #
    # distances, indices = neighbors.kneighbors(query_feature, n_neighbors=max_neighbours)
    #
    # most_similar_image = feature_list[indices[0][0]]
    # print("Most similar image:", most_similar_image)
    # print(indices)



if __name__ == '__main__':
    use_model()
