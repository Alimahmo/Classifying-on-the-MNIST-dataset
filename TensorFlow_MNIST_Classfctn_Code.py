#!/usr/bin/env python
# coding: utf-8

# ### Import the relevant packages
# 
# 

# In[1]:


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds    
# these datasets will be stored in C:\Users\*USERNAME*\tensorflow_datasets\...


# ### Data

# In[2]:


mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
# with_info=True will provide us with a tuple containing information about the version, features, number of samples
# as_supervised=True will load the dataset in a 2-tuple structure (input, target) 
# alternatively, as_supervised=False, would return a dictionary

#extract the training and testing datasets:
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']
# num of validation samples is 10% of the train sample:
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
# let's cast this number to an integer, as a float may cause an error along the way:
num_validation_samples = tf.cast(num_validation_samples, tf.int64)
# accordingly, let's also store the number of test samples in a dedicated variable (instead of using the mnist_info one):
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


# In[3]:


# Normalization: we make our input floats bw 0 and 1:
def scale(image, label):
    # we make sure the value is a float
    image = tf.cast(image, tf.float32)
    # since the possible values for the inputs are 0 to 255 (256 different shades of grey)
    # if we divide each element by 255, we would get the desired result -> all elements will be between 0 and 1 
    image /= 255.

    return image, label

# the method .map() allows us to apply a custom transformation (custom function) to a given dataset
# we have already decided that we will get the validation data from mnist_train, so: 
scaled_train_and_validation_data = mnist_train.map(scale)

# finally, we scale and batch the test data
# we scale it so it has the same magnitude as the train and validation
# there is no need to shuffle it, because we won't be training on the test data
# there would be a single batch, equal to the size of the test data
test_data = mnist_test.map(scale)
#Shuffle train and validation data; Shuffle in batches (here named buffer_size) of 10000:
BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
# Extract validation data with .take method, extract a batch of 10000 observation points = num_validation_samples:
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
# train data set is everything else, with .skip method:
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)
# Batching:
BATCH_SIZE = 100
# for mini-batch GD
train_data = train_data.batch(BATCH_SIZE)
# single batch GD, bc only forward propagation:
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)
validation_inputs, validation_targets = next(iter(validation_data))


# ### Outline the model
# 

# In[4]:


input_size = 784
output_size = 10
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 50
    
# define how the model will look like
model = tf.keras.Sequential([
           tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # input layer
           # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
           # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
           tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
           tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    
           # the final layer is no different, we just make sure to activate it with softmax, bc classification:
           tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])


# ### Choose the optimizer and the loss function
# 

# In[5]:


# Use this to indicate optimizer and loss function:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 'sparse_categorical_crossentropy’ >>> applies one-hot encoding; so that the output and the target layer have matching forms.
# The metric that we wish to be calculated is accuracy.


# ### Training

# In[6]:


# determine the maximum number of epochs
NUM_EPOCHS = 5

# we fit the model, specifying the
# training data
# the total number of epochs
# and the validation data we just created ourselves in the format: (inputs,targets)
model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets),validation_steps=10, verbose =2)


# ### Test the Model

# In[7]:


# To test the model we use .evaluate method
test_loss, test_accuracy = model.evaluate(test_data)


# In[8]:


print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%' .format(test_loss, test_accuracy*100.))


# In[ ]:




