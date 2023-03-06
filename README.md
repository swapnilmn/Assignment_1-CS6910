# Problem Statement

The aim of this task is to develop a feedforward neural network model along with backpropagation algorithm and apply gradient descent (and its variants) for a classification problem. Additionally, we will utilize wandb.ai to keep a record of our experiments and their results.

You can access the task instructions [here](https://wandb.ai/cs6910_2023/A1/reports/CS6910-Assignment-1--VmlldzozNTI2MDc5).


# Installation

Install wandb: 
Install Numpy: 
Install Keras:

# Question 1
  Approch:
    1. Read input data and create an empty dictionary to store class-wise data.
    2. Iterate through the input data and populate the dictionary with images belonging to each class.
    3. Initialize an empty list to store the selected images.
    4. For each class in the dictionary, select the first image and add it to the list of selected images.
    5. Visualize the selected images using library Matplotlib.
    6. Integrate Wandb by initializing a new run, logging the selected images.

The code for question 1 is available at the following [link](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/Assignment_1_Q1.ipynb). Additionally, a WandB visualization for question 1 is accessible at the following [link](https://wandb.ai/ed22s009/Question_1_?workspace=user-ed22s009).

# Question 2
  Approch:
    1. The code contains a class named "OneHotEncoder_from_scratch" for one-hot encoding of categorical variables.
    2. The code defines a class FFNetwork that implements a feedforward neural network for classification tasks.
    3. The network is initialized with hyperparameters such as the number of epochs, the number of hidden layers, the size of each hidden layer, the learning rate, and the activation function.
    4. The weights and biases of the network are initialized using either random or Xavier initialization.
    5. The network has methods for computing the forward activation, gradient of the activation, and softmax function.
    6. The forward_pass method computes the output of the network given an input X and the current weights and biases.
The code for question 1 is available at the following [link](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/Question_2ipynb.ipynb).
