import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  #get training data and test data

"""
Using logits method - more accurate due to lower roundoff error
"""

print(f"Training samples: {len(x_train)}")
print(f"Test samples:     {len(y_train)}")


#image is a shape, but tensorflow expects features to be a row column for neural network
#e.g. [ 1 2
#       3 4] --> [1 2 3 4]

#so flatten image (made into a row column)
x_flat = x_train.reshape(-1, 784)        # -1 --> counts the number of training samples we have, 784 --> number of features per image

#now normalise
norm_layer = tf.keras.layers.Normalization(axis=-1)
norm_layer.adapt(x_flat) # Learn mean and variance
x_normalized = norm_layer(x_flat)

print(f"Max value after: {np.max(x_normalized)}")

#now flatten and normalise test data too
x_test_flat = x_test.reshape(-1, 784)


x_test_normalized = norm_layer(x_test_flat)

print("Test data ready.")


#Build neural network

model = Sequential([
    #input layer (2 features
    tf.keras.Input(shape=(784,)),

    #hidden layer 1 - 25 neurons
    Dense(25, activation='relu', name='layer1'),

    #hidden layer 2 - 15 neurons
    Dense(15, activation='relu', name='layer2'),

    #output layer - 10 possible values now
    Dense(10, activation='linear', name='layer3')   #outputs raw score rather than probabilities
])

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),   #loss function now uses raw logits
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']    #tells it to calculate the accuracy metric as well for later
) #SparseCategoricalCrossentropy is used for softmax, returns one category (in this case 0-9)

#train model
print("Starting training...")
model.fit(
    x_normalized,     # Training Inputs
    y_train,   # Training Labels
    epochs=10,
    batch_size=32,
    verbose=1
)


#check accurary
#TensorFlow automatically handles the logits conversion during evaluation - no need to convert
loss, accuracy = model.evaluate(x_test_normalized, y_test)   #testing the test data we took from MNIST dataset

print("Test Accuracy:", accuracy*100)


#pick out random image from the dataset
index = np.random.randint(0, len(x_test_normalized))

input_image = x_test_normalized[index].numpy()  #used for the prediction (normalised) - convert back to numpy array so reshape can be used
display_image = x_test[index]    #used for displaying image (not normalised)

#reshape (flatten)
input_batch = input_image.reshape(1, 784)

#predict
predictions = model.predict(input_batch)
# CHANGE 3: Convert Logits to Probabilities for human readability
predictions_p = tf.nn.softmax(predictions)


true_label = y_test[index]



# predictions is a list of 10 probabilities, e.g., [0.01, 0.05, 0.90, ...]
# We use argmax to find the INDEX of the highest probability
predicted_label = np.argmax(predictions_p)
probability = np.max(predictions_p)

print(f"\nTrue Label:      {true_label}")
print(f"Predicted Label: {predicted_label}")
print(f"Confidence:      {probability:.4f}")

# Plot
plt.imshow(display_image, cmap='gray')
plt.title(f"True: {true_label}, Pred: {predicted_label}")
plt.axis('off')
plt.show()