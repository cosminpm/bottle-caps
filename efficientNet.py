import math

import numpy as np
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
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

    img_path = r'C:\Users\manolito\Repositories\GitHub\BottleCaps\test1.jpg'

    query_feature = model.predict(preprocess_input(mpimg.imread(img_path))[np.newaxis, ...])
    distances, indices = neighbors.kneighbors(query_feature)

    plt.imshow(mpimg.imread(img_path), interpolation='lanczos')
    plt.xlabel(img_path.split('.')[0] + '_Original Image', fontsize=20)
    plt.show()
    print('********* Predictions ***********')
    similar_images(indices[0], datagen.filenames)


def similar_images(indices, filenames):
    plt.figure(figsize=(15, 10), facecolor='white')
    plotnumber = 1
    for index in indices:
        if plotnumber <= len(indices):
            ax = plt.subplot(2, 4, plotnumber)
            plt.imshow(mpimg.imread(r'C:\Users\manolito\Repositories\GitHub\BottleCaps\training\\'
                                    + filenames[index]), interpolation='lanczos')
            plotnumber += 1
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
