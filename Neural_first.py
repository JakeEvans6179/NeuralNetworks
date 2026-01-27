import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.models import Sequential

"""
Testing manual implementation of neural network compared to tensorflow model
"""

# ==========================================
#Generating dummy data
# ==========================================
def generate_coffee_data(n_samples=500):
    np.random.seed(42)
    # Generate random temperatures between 150 and 285
    temp = np.random.uniform(150, 285, n_samples)
    # Generate random durations between 11 and 16 minutes
    duration = np.random.uniform(11, 16, n_samples)

    X = np.column_stack((temp, duration))
    Y = np.zeros(n_samples)

    # LOGIC: "Good Roast" is a specific region
    # 1. Temp must be between 175 and 260
    # 2. Duration must be between 12 and 15
    # 3. Correlation: Hotter temp needs shorter time (Diagonal cut)
    for i in range(n_samples):
        t = temp[i]
        d = duration[i]

        # Define the "Good Zone"
        if (175 < t < 260) and (12 < d < 15):
            # This math adds the "diagonal" constraint
            # (High temp + High duration = Bad)
            if (d < -0.03 * t + 22):
                Y[i] = 1  # Label 1 = Good Roast

    return X, Y.reshape(-1, 1)


#Create the data
X, Y = generate_coffee_data()



# ==========================================
#Normalisation
# ==========================================
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  #learns mean, variance
X_normalised = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(X_normalised[:,0]):0.2f}, {np.min(X_normalised[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(X_normalised[:,1]):0.2f}, {np.min(X_normalised[:,1]):0.2f}")

#using tensorflow to train and get weights
# ==========================================
#Build neural network
# ==========================================
tf.random.set_seed(1234)
model = Sequential([
    #input layer (2 features
    tf.keras.Input(shape=(2,)),

    #hidden layer
    Dense(3, activation='sigmoid', name='layer1'),

    #output layer
    Dense(1, activation='sigmoid', name='layer2')
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
) #binary crossentropy is logistic regression

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
print("Training neural network")
#train on x_normalised
history = model.fit(X_normalised, Y, epochs=400, verbose=0)
print("Training finished.")

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()    #extract for testing on regular neural network

# ==========================================
# Define the neural network (manual)
# ==========================================

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def dense(a_in, W, b):
    # Vectorized implementation of each layer
    z = np.matmul(a_in, W) + b
    return sigmoid(z)

def sequential(x, W1, b1, W2, b2):
    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)

    return a2
"""
W1 [[-3.9980567  -4.2404466   0.05707902] - columns are the weight values in each unit
 [ 2.426581   -3.3502994  -2.7668033 ]]
 
b1 [-3.4915886  3.769218  -1.8585107] 

W2 [[-25.160501]
 [ 25.090155]
 [-31.01541 ]]

b2 [-3.1926696]

from previous tensorflow script
"""
x_new = np.random.rand(5, 2).astype(np.float32)

manual_prediction = sequential(x_new, W1, b1, W2, b2)

tensor_prediction = model.predict(x_new)

if np.allclose(manual_prediction , tensor_prediction, atol = 1e-7):
    print("Both networks match")




