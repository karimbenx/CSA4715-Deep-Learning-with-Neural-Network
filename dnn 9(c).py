import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import LambdaCallback
from keras import backend as K  # Import Keras backend

# Generate some dummy data for demonstration
train_X = np.random.rand(100, 10)
train_y = np.random.randint(2, size=(100, 1))
val_X = np.random.rand(20, 10)
val_y = np.random.randint(2, size=(20, 1))

# Define a custom callback function to print the activations
def print_activations(epoch, logs):
    # Get activations of the first layer
    layer_output = model.layers[0].output
    # Define a function to get the output of the first layer
    get_activations = K.function([model.input], [layer_output])
    # Get activations for the training data
    activations = get_activations([train_X])[0]
    print(f'Activations after epoch {epoch + 1}: {activations}')

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=10, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the custom callback for printing activations
print_callback = LambdaCallback(on_epoch_end=print_activations)

# Fit the model with training data and use the custom callback
model.fit(train_X, train_y, epochs=10, batch_size=32, validation_data=(val_X, val_y), callbacks=[print_callback])
