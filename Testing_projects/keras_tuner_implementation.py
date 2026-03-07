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

#prepare normalisation layer for use in build function
norm_layer = tf.keras.layers.Normalization(axis=(1, 2)) #this acts on 2D data
#if putting it after flattening layer we use axis = -1
norm_layer.adapt(x_train)   #learn mean and variance

#normalise data

x_train_normalized = norm_layer(x_train)
x_test_normalized = norm_layer(x_test)
x_cv_normalized = norm_layer(x_val)


print("Test data ready.")

'''
#build keras tuner function
def build_model(hp):
    # 1. Define the tuning variables FIRST
    hp_units = hp.Int('units', min_value=64, max_value=256, step=64)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)

    # Use hp.Choice for L2, since lambdas usually jump by multiples of 10
    hp_l2 = hp.Choice('l2_rate', values=[0.0, 0.001, 0.01, 0.1])

    # 2. Build the model using those variables
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),

        # HIDDEN LAYER 1 (Uses the variables)
        tf.keras.layers.Dense(
            units=hp_units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(hp_l2)
        ),
        tf.keras.layers.Dropout(rate=hp_dropout),

        # HIDDEN LAYER 2 (Reuses the EXACT same variables!)
        tf.keras.layers.Dense(
            units=hp_units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(hp_l2)
        ),
        tf.keras.layers.Dropout(rate=hp_dropout),

        # OUTPUT LAYER (Also reusing the same L2 variable)
        tf.keras.layers.Dense(
            10,
            activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(hp_l2)
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model
'''

#build keras tuner function
def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int('units', min_value=64, max_value=256, step=64), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(10, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001))

    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
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
    x_train_normalized, y_train,
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
#Convert Logits to Probabilities for human readability
predictions_p = tf.nn.softmax(predictions)


true_label = y_test[index]



#predictions is a list of 10 probabilities, e.g., [0.01, 0.05, 0.90, ...]
#We use argmax to find the index of the highest probability
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

#plotting learning curve
#Get the settings of best model from the tuner
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

#Build a new model using the best settings
#model starts with random weights
final_model = tuner.hypermodel.build(best_hps)

print("\n--- Starting Final Training for Learning Curve ---")

#Train from scratch and capture the history
#gives graph from 1st epoch to end
history = final_model.fit(
    x_train_normalized,
    y_train,
    epochs=30, #high limit, early stopper will terminate early
    validation_data=(x_cv_normalized, y_val),
    callbacks=[early_stopper],
    verbose=1
)

#plot learning curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Error (Loss) Curve')
plt.legend()
plt.show()



print("\nhyperparameters of best performing model:")
print(f"Neurons (Units): {best_hps.get('units')}")
print(f"Dropout Rate:    {best_hps.get('dropout'):.2f}")

