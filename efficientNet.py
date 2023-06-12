import math

import numpy as np
from PIL import Image
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

import shutil
import os


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


def main():
    max_neighbours = 10
    img_size = 224
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3), pooling='max')
    batch_size = 64

    root_dir = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\training'

    img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    datagen = img_gen.flow_from_directory(root_dir,
                                          target_size=(img_size, img_size),
                                          batch_size=batch_size,
                                          class_mode=None,
                                          shuffle=False)

    num_images = len(datagen.filenames)
    num_epochs = int(math.ceil(num_images / batch_size))

    feature_list = model.predict_generator(datagen, num_epochs)
    print("Num images   = ", len(datagen.classes))
    print("Shape of feature_list = ", feature_list.shape)

    model.compile()
    save_model(model=model, path=r'C:\Users\manolito\Repositories\GitHub\BottleCaps\model')
    # similar_images(model, feature_list, datagen.filenames, max_neighbours)


def use_model():
    path = r"C:\Users\manolito\Repositories\GitHub\BottleCaps\model"
    img_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\9.jpg'
    max_neighbours = 10
    img_size = 224
    batch_size = 64


    model = load_model(path)
    img = Image.open(img_path)
    resized_img = img.resize((224, 224))  # Resize the image to (224, 224)
    resized_img = np.array(resized_img)
    preprocessed_img = preprocess_input(resized_img[np.newaxis, ...])  # Preprocess the resized image
    query_feature = model.predict(preprocessed_img)

    root_dir = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\training'

    img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    datagen = img_gen.flow_from_directory(root_dir,
                                          target_size=(224, 224),
                                          batch_size=batch_size,
                                          class_mode=None,
                                          shuffle=False)

    num_images = len(datagen.filenames)
    num_epochs = int(math.ceil(num_images / batch_size))

    feature_list = model.predict_generator(datagen, num_epochs)
    print("Num images = ", len(datagen.classes))
    print("Shape of feature_list = ", feature_list.shape)

    neighbors = NearestNeighbors(algorithm='ball_tree', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors(query_feature, n_neighbors=max_neighbours)

    most_similar_image = datagen.filenames[indices[0][0]]
    print("Most similar image:", most_similar_image)

    # ...


def similar_images(model, feature_list, filenames, max_neighbours):
    plt.figure(figsize=(15, 10), facecolor='white')
    plotnumber = 1

    rows = 2  # Number of rows in the subplot grid
    cols = 5  # Number of columns in the subplot grid

    img_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\9.jpg'

    neighbors = NearestNeighbors(algorithm='ball_tree', metric='euclidean')
    neighbors.fit(feature_list)

    query_feature = model.predict(preprocess_input(mpimg.imread(img_path))[np.newaxis, ...])
    distances, indices = neighbors.kneighbors(query_feature, n_neighbors=max_neighbours)

    plt.imshow(mpimg.imread(img_path), interpolation='lanczos')
    plt.xlabel(img_path.split('.')[0] + '_Original Image', fontsize=20)
    plt.show()
    print('********* Predictions ***********')

    for i in range(max_neighbours):
        ax = plt.subplot(rows, cols, plotnumber)
        plt.imshow(
            mpimg.imread(r'C:\Users\manolito\Repositories\GitHub\BottleCaps\training\\' + filenames[indices[0][i]]),
            interpolation='lanczos')
        plotnumber += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # create_training_folder()
    # main()
    use_model()
