import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import os
import sys
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split 
#the train_test_split function from the model_selection module in scikit-learn (sklearn)
EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.3

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop',
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels) #convert labels into one-hot encoded vector, chuyen doi du lieu phan loai thanh du lieu so
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    ) 
    #these are converted into a NumPy array

    if len(sys.argv) == 3:
        # Load the pre-trained model from the file
        model = keras.models.load_model(sys.argv[2])
        print(f"Model loaded from {sys.argv[2]}")
    else:
        # Get a compiled neural network
        model = get_model()

        # Fit model on training data
        history = model.fit(x_train, y_train, epochs=EPOCHS)
        #This function trains the model on the provided training data.
        training_loss = history.history['loss']
        accuracy = history.history['accuracy']
        # retrieves the training values from the history object
        epoch_count = range(1,len(training_loss)+1)
        plt.figure(0)
        plt.plot(epoch_count,training_loss,'r--')
        plt.plot(epoch_count,accuracy,'b--')
        plt.legend(['training loss','accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('history')
        plt.show(block=False)
        plt.pause(60)

        print("Training completed.")

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    if len(sys.argv) != 3:
        # Save model to file
        filename = 'model.h5'
        model.save(filename)
        print(f"Model saved to {filename}.")
    
    test(model)
    

    



def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    # Path to data folder
    data_path = os.path.join(data_dir)

    # Number of subdirectories/labels
    number_of_labels = 0
    
    for i in os.listdir(data_path):
        number_of_labels += 1

    # Loop through the subdirectories
    for sub in range(number_of_labels):
        sub_folder = os.path.join(data_path, str(sub))

        images_in_subfolder = []

        for image in os.listdir(sub_folder):
            images_in_subfolder.append(image)

        # Open each image 
        for image in images_in_subfolder:

            image_path = os.path.join(data_path, str(sub), image)

            # Add Label
            labels.append(sub)

            # Resize and Add Image
            img = cv2.imread(image_path)
            # print(image_path)
            res = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation= cv2.INTER_AREA)
            images.append(res)

    return (np.array(images), np.array(labels))


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([

        # 2 Convolutional layers and 2 Max-pooling layers
        tf.keras.layers.Conv2D(
            40, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            40, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # # Hidden Layers
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        
        # Dropout
        tf.keras.layers.Dropout(0.4),

        # Extra hidden layer
        tf.keras.layers.Dense(128, activation="relu"),
        
        # Output layer with output units for all digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def test(model):
    image_path = "e7edf480-66d9-11eb-97da-c6d7da1a2a9b.jpg"
    test_image = Image.open(image_path).convert("RGB")
    test_image = test_image.resize((IMG_WIDTH, IMG_HEIGHT))
    test_image = np.array(test_image)
    test_image = np.reshape(test_image, (1, IMG_WIDTH, IMG_HEIGHT, 3))
    prediction = model.predict(test_image)
    predicted_label = np.argmax(prediction)
    NumberElement = prediction.argmax()
    plt.figure(1)
    plt.imshow(test_image[0], cmap='gray')  # Access the first image in the batch
    plt.title(classes[NumberElement])
    plt.show(block=False)
    plt.pause(30)
    print(f"Predicted label: {classes[NumberElement]}")
    plt.show()



if __name__ == '__main__':
    main()
