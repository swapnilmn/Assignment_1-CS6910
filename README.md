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

The code for question 1 is available at the following [link](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/Assignment_1_Q1.ipynb) . Additionally, a WandB visualization for question 1 is accessible at the following [link](https://wandb.ai/ed22s009/Question_1_?workspace=user-ed22s009).
