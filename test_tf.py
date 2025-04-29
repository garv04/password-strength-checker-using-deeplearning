import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Create a simple model
model = Sequential([
    layers.Dense(10, activation='relu', input_shape=(5,)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
print("Model created successfully!") 