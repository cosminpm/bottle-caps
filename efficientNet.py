import math

import numpy as np
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


def main():
    max_neighbours = 30
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

    neighbors = NearestNeighbors(algorithm='ball_tree', metric='euclidean')
    neighbors.fit(feature_list)

    img_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\9.jpg'

    query_feature = model.predict(preprocess_input(mpimg.imread(img_path))[np.newaxis, ...])
    distances, indices = neighbors.kneighbors(query_feature, n_neighbors=max_neighbours)

    plt.imshow(mpimg.imread(img_path), interpolation='lanczos')
    plt.xlabel(img_path.split('.')[0] + '_Original Image', fontsize=20)
    plt.show()
    print('********* Predictions ***********')
    similar_images(indices[0], datagen.filenames, max_neighbours)


def similar_images(indices, filenames, max_neighbours):
    plt.figure(figsize=(15, 10), facecolor='white')
    plotnumber = 1

    rows = 3  # Number of rows in the subplot grid
    cols = 10  # Number of columns in the subplot grid

    for i in range(max_neighbours):
        ax = plt.subplot(rows, cols, plotnumber)
        plt.imshow(mpimg.imread(r'C:\Users\manolito\Repositories\GitHub\BottleCaps\training\\' + filenames[indices[i]]), interpolation='lanczos')
        plotnumber += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #create_training_folder()
    main()