import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.models import Sequential

"""
AI generated - tensorflow implementation on dummy coffee data
"""

# ==========================================
# 1. GENERATE DUMMY "COFFEE" DATA
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


# Create the data
X, Y = generate_coffee_data()

# Plot the raw data first to see what we are dealing with
print("Data generated. Red X = Good Roast, Blue O = Bad Roast.")

# ==========================================
# 2. PREPROCESSING (Normalization)
# ==========================================

#normalize data to be roughly -1 to 1.
print(f"Raw Temp Range: {np.min(X[:, 0]):.1f} to {np.max(X[:, 0]):.1f}")

norm_layer = Normalization(axis=-1)
norm_layer.adapt(X)  # Learn the mean/variance of our data
X_normalized = norm_layer(X)  # Create a normalized version for training

# ==========================================
# 3. BUILD THE NEURAL NETWORK
# ==========================================
tf.random.set_seed(1234)
model = Sequential([
    # Input Layer (2 features: Temp, Duration)
    tf.keras.Input(shape=(2,)),

    # Hidden Layer (3 Neurons) -> The "Scouts"
    # They will find the boundaries (Too hot, Too cold, Too long)
    Dense(3, activation='sigmoid', name='layer1'),

    # Output Layer (1 Neuron) -> The "General"
    # Combines the scouts' info to say Yes/No
    Dense(1, activation='sigmoid', name='layer2')
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
)

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
print("Training the neural network... (This takes a few seconds)")
# We train on X_normalized, NOT X!
history = model.fit(X_normalized, Y, epochs=400, verbose=0)
print("Training finished.")

# ==========================================
# 5. VISUALIZE THE DECISION BOUNDARY
# ==========================================
# Create a grid of points covering the whole graph
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Flatten the grid so we can feed it to the model
grid_points = np.c_[xx.ravel(), yy.ravel()]

# IMPORTANT: We must normalize the grid points too!
# The model only understands "normalized" numbers now.
grid_preds = model.predict(norm_layer(grid_points))
Z = grid_preds.reshape(xx.shape)

# Plot everything
plt.figure(figsize=(10, 6))

# Draw the decision boundary (The "Safe Zone" in Red/Warm color)
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="coolwarm_r", alpha=0.3)

# Scatter plot of the original data
# Bad Roasts (0) in Blue
plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c='blue', marker='o', label="Bad Roast")
# Good Roasts (1) in Red
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c='red', marker='x', label="Good Roast")

plt.title("Neural Network Decision Boundary (Coffee Roasting)")
plt.xlabel("Temperature (Celsius)")
plt.ylabel("Duration (Minutes)")
plt.legend()
plt.show()



# 1. Get the weights from the trained layers
W1, b1 = model.get_layer("layer1").get_weights()
print("W1", W1)
print("b1", b1)
W2, b2 = model.get_layer("layer2").get_weights()
print("W2", W2)
print("b2", b2)

# 2. Get the normalization stats (Crucial!)
# Your manual script needs to normalize raw data exactly like TF did.
norm_mean = norm_layer.mean.numpy()
norm_variance = norm_layer.variance.numpy()
print("Normalization mean:", norm_mean)
print("Normalization variance:", norm_variance)

print("Copied weights and normalization stats!")
