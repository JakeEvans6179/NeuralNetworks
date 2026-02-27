import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Normalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

# Load Fashion MNIST instead of standard MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(f"training examples: {len(x_train)}")
print(f"test examples: {len(x_test)}")

#flatten images
xtrain_flat = x_train.reshape(-1, 784)
xtest_flat = x_test.reshape(-1, 784)

#run normalisation
norm_layer = tf.keras.layers.Normalization(axis=-1)
norm_layer.adapt(xtrain_flat) # Learn mean and variance
x_normalized = norm_layer(xtrain_flat)


x_test_normalized = norm_layer(xtest_flat)

print("Test data ready.")

model = Sequential([
    #input layer (2 features
    tf.keras.Input(shape=(784,)),

    #hidden layer 1 - 25 neurons
    Dense(256, activation='relu', name='layer1'),
    Dropout(0.2),  # <--- NEW: Randomly turns off 20% of the 256 neurons each batch

    #hidden layer 2 - 15 neurons
    Dense(128, activation='relu', name='layer2'),
    Dropout(0.2), #randomly turns off 20% of the 128 neurons each batch (improve generalisation)

    #output layer - 10 possible values now
    Dense(10, activation='linear', name='layer3')   #outputs raw score rather than probabilities
])

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),   #loss function now uses raw logits
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']    #tells it to calculate the accuracy metric as well for latqer
) #SparseCategoricalCrossentropy is used for softmax, returns one category (in this case 0-9)

#train model
print("Starting training...")
model.fit(
    x_normalized,     # Training Inputs
    y_train,   # Training Labels
    epochs=25,
    batch_size=32,
    verbose=1
)

#test testing dataset accuracy
loss, accuracy = model.evaluate(x_test_normalized, y_test)

print(f"testing accuracy: {accuracy*100}")







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

#now gap between training and test accuracy is decreasing, meaning memorisation has decreased and model generalises better
