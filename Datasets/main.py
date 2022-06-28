import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# See all registered datasets
builders = tfds.list_builders()
print(builders)

# Load a given dataset by name, along with the DatasetInfo metadata
data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data["train"], data["test"]

print(info)

# Creating datasets from numpy array
num_items = 100
num_list = np.arange(num_items)

# creating the dataset from numpy array
num_list = tf.data.Dataset.from_tensor_slices(num_list)

# Creation
# 1. via from_tensor_slices() => accepts individual NumPy (or tensors) and batches
# 2. via from_tensors() => save as 1. but does not support batches
# 3. via from_generator() => takes input from a generator frnction

# Transformation:
# 1. via batch() => sequentially divides the dataset by the specified size
# 2. via repeat() => duplicated the data
# 3. via shuffle() => randomly shuffles the data
# 4. via map() => applies a function to the data
# 5. filter() => applies a filter function to the data

# Iterators:
# via next_batch = iterator.get_next()
