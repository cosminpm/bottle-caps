import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

from ScriptsMain.utilsFun import read_img_from_path

NUM_CLASSES = 110
EMBEDDING_SIZE = 64
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.001

def create_folder():
    path = r'C:\Users\cosmi\Desktop\BottleCaps\database\caps-resized'
    images = os.listdir(path)

    training_folder = r'C:\Users\cosmi\Desktop\BottleCaps\training'
    for image in images:
        sub_folder_name = image.split(".")[0]
        path_create = os.path.join(training_folder, sub_folder_name)
        os.mkdir(path_create)

        actual_path = os.path.join(path, image)
        destiny_image = os.path.join(training_folder, sub_folder_name, image)
        os.popen(f'copy {actual_path} {destiny_image}')



def main():
    # Constants


    # Load pre-trained EfficientNet-B0 model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Add custom layers for embedding and classification
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    embedding_layer = Dense(EMBEDDING_SIZE, activation='relu')(x)
    output_layer = Dense(NUM_CLASSES, activation='softmax')(embedding_layer)

    # Create the model
    model = Model(inputs=base_model.input, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss=CategoricalCrossentropy(),
                  metrics=[Accuracy()])

    # Load and preprocess the data
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        r'C:\Users\cosmi\Desktop\BottleCaps\training',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training')

    val_generator = datagen.flow_from_directory(
        r'C:\Users\cosmi\Desktop\BottleCaps\training',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation')

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=NUM_EPOCHS)

    tf.saved_model.save(model, r'C:\Users\cosmi\Desktop\BottleCaps\model')


def resize_image(image_path, target_size):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Resize the image using the target size
    resized_image = cv2.resize(image, target_size)

    # Return the resized image
    return resized_image


def preprocess_image(image_path, target_size):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Resize the image using the target size
    resized_image = cv2.resize(image, target_size)

    # Preprocess the image using the EfficientNet preprocessing function
    preprocessed_image = tf.keras.applications.efficientnet.preprocess_input(resized_image)

    # Convert the image to float32 dtype
    preprocessed_image = preprocessed_image.astype(np.float32)

    # Add a batch dimension
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    # Return the preprocessed image
    return preprocessed_image




def test_model():
    loaded_model = tf.saved_model.load(r'C:\Users\cosmi\Desktop\BottleCaps\model')

    IMAGE_SIZE = (256, 256)
    image_path = r'C:\Users\cosmi\Desktop\BottleCaps\database\caps-resized\cap-3_100.jpg'

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path, IMAGE_SIZE)

    # Make predictions using the loaded model
    predictions = loaded_model(preprocessed_image)

    # Process the predictions
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_probability = predictions[0][predicted_class].numpy()

    # Load and preprocess the data
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        r'C:\Users\cosmi\Desktop\BottleCaps\training',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training')

    # Get the class labels
    class_labels = train_generator.class_indices

    # Find the name of the best match class label
    best_match_name = None
    for name, index in class_labels.items():
        if index == predicted_class:
            best_match_name = name
            break

    # Print the result
    print("Best Match Class Label:", best_match_name)
    print("Probability:", predicted_probability)


if __name__ == '__main__':
    #print_all_classes()
    #main()
    test_model()
    #create_folder()