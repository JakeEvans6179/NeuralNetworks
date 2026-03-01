import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Normalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.model_selection import train_test_split

'''
Using keras tuner we can alter the number of neurons per layer as well as the dropout rate and run search on best combination
Randomised 5 searches to save time, with many combinations e.g. 10,000 running 30-50 randomised searches are sufficient to find a good model

Added early stoppage in order to stop training once the validation set is not improved for 3 epochs in a row, taking the weights of the best epoch as 
final weights

Saved final model onto disk for reuse 
'''

#load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

#slice training data into training set (50k) and cross validation set (10k)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=10000, random_state=42
)

print(f"training examples: {len(x_train)}")
print(f"test examples: {len(x_test)}")



#run normalisation
norm_layer = tf.keras.layers.Normalization(axis=-1)
norm_layer.adapt(x_train) # Learn mean and variance
x_normalized = norm_layer(x_train)

x_cv_normalized = norm_layer(x_val)

x_test_normalized = norm_layer(x_test)

print("Test data ready.")

#build keras tuner function


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Let the Tuner pick the number of Neurons (between 64 and 256, in steps of 64)
    hp_neurons = hp.Int('units', min_value=64, max_value=256, step=64)
    model.add(tf.keras.layers.Dense(units=hp_neurons, activation='relu'))

    # Let the Tuner pick the Dropout rate (between 0.1 and 0.5)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(rate=hp_dropout))

    model.add(tf.keras.layers.Dense(10, activation='linear'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


#set up early stopper --> stop when model cross validation accuracy doesn't improve after 3 epochs in a row
early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,     #only stop training if validation accuracy doesn't improve 3 epochs in a row
    restore_best_weights=True   #uses weights of best epoch as final weights
)

#start tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  #It will test 5 completely random combinations of Neurons + Dropout, for large combinations e.g. 10,000, 30-50 random samples is usually enough
    directory='my_tuning_dir',
    project_name='fashion_dense_tuning'
)

#start search
print("Starting KerasTuner Search...")
tuner.search(
    x_normalized, y_train,
    epochs=20,  #training rounds
    validation_data=(x_cv_normalized, y_val),
    callbacks=[early_stopper]
)

#get best model and run accuracy test on test dataset for estimated generalisation accuracy
best_model = tuner.get_best_models(num_models=1)[0]
print("\n--- TUNING COMPLETE ---")
print("Evaluating best model on test dataset")

test_loss, test_accuracy = best_model.evaluate(x_test_normalized, y_test)
print(f"Final True Generalization Accuracy: {test_accuracy * 100:.2f}%")


# Save the fully trained winner to a single, portable file
best_model.save('my_ultimate_fashion_model.keras')




#pick out random image from the dataset
index = np.random.randint(0, len(x_test_normalized))

input_image = x_test_normalized[index].numpy()  #used for the prediction (normalised) - convert back to numpy array so reshape can be used
display_image = x_test[index]    #used for displaying image (not normalised)

#Put the image into a "batch of 1" --> keras can only look at a batch of images
input_batch = input_image.reshape(1, 28, 28)
#predict
predictions = best_model.predict(input_batch) #no need to flatten image as NN does it in pipeline
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