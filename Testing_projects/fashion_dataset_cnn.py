import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

#Load Data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(f"Training examples: {len(x_train)}")
print(f"Test examples:     {len(x_test)}")

#Preprocessing for CNNs (keep 2D)
#Instead of flattening to 784, we keep the 28x28 grid.
x_train_cnn = x_train.reshape(-1, 28, 28, 1) #keep 2D, 1 at end tells CNN it is greyscale
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

#normalise images
norm_layer = tf.keras.layers.Normalization(axis=-1)
norm_layer.adapt(x_train_cnn) # Learn mean and variance
x_train_normalized = norm_layer(x_train_cnn)

x_test_normalized = norm_layer(x_test_cnn)


print("Data ready for CNN.")

#Build CNN
model = Sequential([
    # Input layer: 28x28 pixels, 1 color channel
    tf.keras.Input(shape=(28, 28, 1)),

    #Feature extraction (CNN)
    #32 magnifying glasses (3x3) looking for edges --> fewer filters in beginning for simple features, more afterwards for higher level abstract features
    Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv1'),
    #Shrink the image to save memory and highlight the best features
    MaxPooling2D(pool_size=(2, 2), name='pool1'),

    #64 magnifying glasses looking for complex shapes (like collars/shoelaces)
    Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
    MaxPooling2D(pool_size=(2, 2), name='pool2'),

    #Classification (Dense network)
    #Flatten it into a 1D line.
    Flatten(),

    #Dense hidden layer to process the features
    Dense(128, activation='relu', name='dense1'),
    #Deactivate 20% of neurons to prevent memorization
    Dropout(0.2),

    #Output layer: 10 raw scores for the 10 clothing categories
    Dense(10, activation='linear', name='output')
])

#Compile
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

#Train
print("Starting CNN training...")
model.fit(
    x_train_normalized,
    y_train,
    epochs=10,
    batch_size=32,
    verbose=1
)

# 6. Evaluate
loss, accuracy = model.evaluate(x_test_normalized, y_test)
print(f"\nFinal CNN Test Accuracy: {accuracy * 100:.2f}%")