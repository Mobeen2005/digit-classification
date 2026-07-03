
"""
MNIST Handwritten Digit Classification using CNN
Author: Mohammad Mobeen
Description: Builds, trains, and evaluates a Convolutional Neural Network (CNN)
on the MNIST dataset to classify handwritten digits (0-9).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """Load the MNIST dataset and preprocess it for training."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    ''' Reshape to (samples, height, width, channels) and normalize pixel values to [0, 1] '''
    
    x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

    ''' One-hot encode the labels '''
    
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Build and return a CNN model for digit classification."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_training_history(history):
    """Plot training and validation accuracy over epochs."""
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Model Accuracy over Epochs')
    plt.savefig('training_accuracy.png')
    plt.show()


def main():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_model()
    model.summary()

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')

    plot_training_history(history)


if __name__ == '__main__':
    main()
