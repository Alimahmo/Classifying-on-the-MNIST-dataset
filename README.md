# Classifying-on-the-MNIST-dataset
Using Machine Learning with TensorFlow 2.0 to solve a classification problem; 
The problem is to classify handwritten digits from a dataset of 70000 images.
This dataset is a default in TF2-datasets, and it is known as MNIST dataset.
MNIST Classification is considered the "Hello World!" of Machine Learning.

# Description:

Classification problem. Targets are numbers from 0 to 9.
Each image 28*28 pixels. 784 pixels. It is on greyscale [0,255].
The approach for deep feed forward neural networks is to transform or “flatten” each image into a vector of length 784*1.
Our Input Layer has 784 Input Nodes.

We choose 2 hidden layers. 
Output Layer: 10 nodes: 10 digits: 10 classes
We use One-hot Encoding for the Outputs to be compared to Targets.
Activation Function = Softmax
![image](https://user-images.githubusercontent.com/80431527/144125496-5f734daa-bd4e-43fb-874b-e5e1cf43b27c.png)


#
Train >> Validation >> Test
: This is actually what ML is about
#

# The MNIST action plan.
1. Prepare the data and preprocess it. Create training, validation and test datasets.
2. Outline the model and choose the activation functions.
3. Set the appropriate advance optimizers and loss function.
4. Train the model.
5. Test the accuracy of the model.


TF Datasets: TFDS:
https://www.tensorflow.org/datasets/catalog/mnist
TFDS has already split the dataset into “Training” and “Test” datasets. 


Install TFDS Package using:
/pip install tensorflow-datasets 
 or
/conda install tensorflow-datasets
