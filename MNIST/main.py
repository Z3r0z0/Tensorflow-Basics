import tensorflow as tf
from tensorflow import keras

# Network and training parameters
EPOCHS = 20  # defines how long the training should last
BATCH_SIZE = 128  # is the numer of samples which are fed in to the network at a time
VERBOSE = 1
NB_CLASSES = 10  # number of outputs == number of digits
N_HIDDEN = 128  # number of hidden nodes
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3  # random dropout rate for values

# Loading MNIST dataset
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# X_train is 60000 rows of 28x28 values;
# reshape to 60000 x 784
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Normalize inputs to be within in [0, 1]
X_train /= 255
X_test /= 255

print(X_train.shape[0], "train samples")
print(Y_train.shape[0], "test samples")

# One-hot representation of the labels
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# Building the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPED,), name="dense_layer_1", activation="relu"))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN, name="dense_layer_2", activation="relu"))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_CLASSES, name="dense_layer_3", activation='softmax'))

# Summary of the model
model.summary()

# Compiling the model
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training the model
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Evaluating the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy: ", test_acc)
