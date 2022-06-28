import tensorflow as tf

# Create Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, input_shape=(784,), name="dense_layer_1", activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(100, name="dense_layer_2", activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(10, name="dense_layer_3", activation='softmax'))

# Summary of the model
model.summary()

# Compiling the model
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Save weights to a Tensorflow Checkpoint file
model.save_weights("./checkpoint_save/weights_save")

# Save weights to a HDF5 file
model.save_weights("./hdf5_save/weights_save.h5", save_format="h5")

# Load weights
model.load_weights("./checkpoint_save/weights_save")
model.load_weights("./hdf5_save/weights_save.h5")


# JSON serialization
json_string = model.to_json()  # save
model = tf.keras.models.model_from_json(json_string)  # restore


# Saving a model with weights and the optimization parameters
model.save("./full_save/my_model.h5")
model = tf.keras.models.load_model("./full_save/my_model.h5")
