import tensorflow as tf

# Network and training parameters
EPOCHS = 20  # defines how long the training should last
BATCH_SIZE = 128  # is the numer of samples which are fed in to the network at a time
VERBOSE = 1
NB_CLASSES = 10  # number of outputs == number of digits
N_HIDDEN = 128  # number of hidden nodes
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3  # random dropout rate for values

RESHAPED = 784

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPED,), name="dense_layer_1", activation="relu"))
model.add(tf.keras.layers.Dropout(DROPOUT))
model.add(tf.keras.layers.Dense(N_HIDDEN, name="dense_layer_2", activation="relu"))
model.add(tf.keras.layers.Dropout(DROPOUT))
model.add(tf.keras.layers.Dense(NB_CLASSES, name="dense_layer_3", activation='softmax'))
