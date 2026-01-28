import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  #get training data and test data

#apply filter to just get binary 1 and 0 (binary classification task)

mask = (y_train == 0) | (y_train == 1)      #gets only 1 and 0 images and puts onto list x_train_binary and y_train_binary
x_train_binary = x_train[mask]
y_train_binary = y_train[mask]

#mask for test values
test_mask = (y_test == 0) | (y_test == 1)

# Apply mask
x_test_binary = x_test[test_mask]
y_test_binary = y_test[test_mask]

print(f"Training samples (0s and 1s): {len(y_train_binary)}")
print(f"Test samples (0s and 1s):     {len(y_test_binary)}")


#image is a shape, but tensorflow expects features to be a row column for neural network
#e.g. [ 1 2
#       3 4] --> [1 2 3 4]

#so flatten image (made into a row column)
x_flat = x_train_binary.reshape(-1, 784)        # -1 --> counts the number of training samples we have, 784 --> number of features per image

#now normalise
norm_layer = tf.keras.layers.Normalization(axis=-1)
norm_layer.adapt(x_flat) # Learn mean and variance
x_normalized = norm_layer(x_flat)

print(f"Max value after: {np.max(x_normalized)}") # Should be roughly > 0 (Standard Deviation)

#now flatten and normalise test data too
x_test_flat = x_test_binary.reshape(-1, 784)


x_test_normalized = norm_layer(x_test_flat)

print("Test data ready.")


#Build neural network

model = Sequential([
    #input layer (2 features
    tf.keras.Input(shape=(784,)),

    #hidden layer 1 - 25 neurons
    Dense(25, activation='sigmoid', name='layer1'),

    #hidden layer 2 - 15 neurons
    Dense(15, activation='sigmoid', name='layer2'),

    #output layer
    Dense(1, activation='sigmoid', name='layer3')
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    metrics=['accuracy']    #tells it to calculate the accuracy metric as well for later
) #binary crossentropy is logistic regression

#train model
print("Starting training...")
model.fit(
    x_normalized,     # Training Inputs
    y_train_binary,   # Training Labels
    epochs=10,
    batch_size=32,
    verbose=1
)


#check accurary
loss, accuracy = model.evaluate(x_test_normalized, y_test_binary)

print("Test Accuracy:", accuracy*100)


#pick out random image from the dataset
index = np.random.randint(0, len(x_test_normalized))

input_image = x_test_normalized[index].numpy()  #used for the prediction (normalised) - convert back to numpy array so reshape can be used
display_image = x_test_binary[index]    #used for displaying image (not normalised)

#reshape (flatten)
input_batch = input_image.reshape(1, 784)

#predict
prediction_prob = model.predict(input_batch)
prediction_value = prediction_prob[0][0]  #probability of being 1


if prediction_value >= 0.5:
    label = "Predicted 1"
else:
    label = "Predicted 0"

# 5. Show the result
print("Prediction probability:", prediction_value)  #probability of being a 1
print("Prediction value:", label)

#plot image
plt.imshow(display_image, cmap='gray')
plt.show()