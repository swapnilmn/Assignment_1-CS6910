# Problem Statement

The aim of this task is to develop a feedforward neural network model along with backpropagation algorithm and apply gradient descent (and its variants) for a classification problem. Additionally, we will utilize wandb.ai to keep a record of our experiments and their results.

You can access the task instructions [here](https://wandb.ai/cs6910_2023/A1/reports/CS6910-Assignment-1--VmlldzozNTI2MDc5).


# Installation

Install wandb: 
Install Numpy: 
Install Keras:

# Question 1
Approch:
1. Read input data and created an empty dictionary to store class-wise data.
2. Iterated through the input data and populated the dictionary with images belonging to each class as representative.
3. Initialized an empty list to store the selected images from each class of data.
4. For each class in the dictionary, selected the first image and added it to the list of selected images.
5. Visualized the selected images using library Matplotlib for both Train data and Test data.
6. Integrated Wandb by initializing a new run, logging the selected images And pasted those wandb images into wandb report.

The code for question 1 is available at the following [link](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/Assignment_1_Q1.ipynb). Additionally, a WandB visualization for question 1 is accessible at the following [link for Test data](https://wandb.ai/ed22s009/Question_4_Best_Model/reports/Test-Sample-Images-23-03-07-09-46-44---VmlldzozNzE5NDE2?accessToken=5pjoktdiyt55kxfa5ice07170c49t7q51nxsg94urfftg7sqe0lnwlushquvy5el), [link for train data](https://wandb.ai/ed22s009/Question_4_Best_Model/reports/Train-Sample-Images-23-03-07-09-47-46---VmlldzozNzE5NDIz?accessToken=zlhk92dggxsrfawk6vl1sm9ctbeqsng9q8bqm7vvt4r1weaee0pyqodwk4xge1l1).

# Question 2
Approch:
1. For one-hot encoding of categorical variables class named "OneHotEncoder_from_scratch" is created and labels are converted into one hot matrix
2. Class FFNetwork defined that implementes a feedforward neural network for classification tasks of images
3. The network is initialized with hyperparameters such as the number of epochs, the number of hidden layers, the size of each hidden layer, the learning rate, and the activation function.
4. The weights and biases of the network are initialized using either random or Xavier initialization.
5. The network has methods for computing the forward activation, gradient of the activation, and softmax function.
6. The forward_pass method computes the output of the network given an input X and the current weights and biases. It prints outputs as probability distribution over the 10 classes.
7. You can see the code is enough flexible to change number of hidden layers and correspondingly repsectlively number of neurons in each hidden layers. Just changing values when calling class will give changes. Both number of hidden layers and neurons in each layer of hidden layers can be changed

The code for question 2 is available at the following [link](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/Question_2ipynb.ipynb).

# Question 3
Approch:
1. Addition to last question backpropogation algorithm added, that performs backpropagation to calculate gradients for each layer of the neural network. 
2. The method first performs a forward pass using the input data and current weights. It then computes the derivative of the loss (For both Cross entropy and Mean square error) with respect to the final activation layer. Using this derivative, the method iteratively computes the derivatives for each previous layer, propagating backwards through the network.
3. L2 Regularisation is added in backprop only to not to overfit model
4. Finally, it returns the derivative of the loss with respect to the input layer activations.
5. The fit method takes input parameters such as X and Y training data, X_val and Y_val validation data, and several hyperparameters such as the optimization algorithm, learning rate, etc.
6. The method trains the neural network for the given number of epochs using a loop and prints the train loss, train accuracy, validation loss, and validation accuracy every five epochs.
7. The method implements four optimization algorithms (SGD, Momentum, RMSProp, Adam) and updates the weights and biases of the neural network in each epoch accordingly.
8. It prints the training and validation loss and accuracy at each epoch.

The code for question 3 is available at the following [link](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/Question_3.ipynb).

# Question 4
Approch:
1. Initializes Weights and Biases (WandB) to track and visualize training progress.Defines a configuration dictionary containing hyperparameters such as learning rate, number of epochs, optimizer, etc.
2. Defines a train function that initializes WandB with the given configuration, sets up the neural network model, trains the model, and generates a confusion matrix plot.
3. The neural network model is defined using a FFNetwork class, which takes in the given configuration and trains the model using various optimization algorithms such as SGD, Momentum, Adam, etc.
4. The sweep configuration includes the method, name, metric, and parameters for the sweep. The metric used to evaluate the sweep results is val_accuracy, and the goal is to maximize this metric.
5. The parameters in the sweep include epochs, hidden_layer_count, size_hidden_layers, learning_rate, optimizer, batch_size, activation, weight_initializations, and weight_decay. The sweep is created using the sweep_config and run using the wandb.agent() function with the train() function.

The code for question 4 is available at the following [link](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/Assignment_1_Question4.ipynb). Additionally, a WandB visualization for question 4 is accessible at the following [link](https://wandb.ai/ed22s009/Question_4_Best_Model/reports/Question-4--VmlldzozNzA5ODcx).

# Question 5
1. Best Accuracy:  89.53  (Validation), 93.21 (Training)

        sweep_config = {
                'method': 'grid',
                'name': 'Assignement1',
                'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
                'parameters': {
                  'epochs': {'values': [18]},#150
                  'hidden_layer_count':{'values': [3]},
                  'size_hidden_layers':{'values': [256]},
                  'learning_rate':{'values': [0.001]},
                  'optimizer':{'values': ['nadam']},
                  'batch_size':{'values': [128]},
                  'activation':{'values': ['tanh']},
                  'weight_initializations':{'values': ['Xavier']},
                  'weight_decay':{'values': [0]}}
                }

The wandb visualisation for question 5 can be found at this [link](https://api.wandb.ai/links/ed22s009/mcze8md4).

# Question 6
The wandb visualisation for question 6 can be found at this [link](https://api.wandb.ai/links/ed22s009/01bhldhw).

# Question 7
Approch:

1. The approach to finding the best machine learning model involves experimenting with different hyperparameters and configurations to find the best combination.
2.  Once the best model is identified, its accuracy and other relevant metrics should be reported. 
3.  Implementation of  a function to calculate confusion matrix done.
4.  Parameter for best model:

        configuration = {
            'learning_rate': 0.001,
            'epochs': 18,
            'hidden_layer_count': 3,
            'size_hidden_layers': 256,
            'optimizer': 'nadam',
            'batch_size': 128,
            'activation': 'tanh',
            'weight_initializations': 'Xavier',
            'weight_decay': 0,
            'loss_function': 'cross_entropy',
            'dataset': 'fashion_mnist'
        }
        
 5. Val Accuracy is 89.53 for this config.
 6.  Plot and integrate wandb to keep track using wandb. The code for question 7 can be found here [link](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/Assignment_1_Question7.ipynb). The wandb visualisation for question 7 can be found here [link](https://wandb.ai/ed22s009/Question_4_Best_Model/reports/undefined-23-03-07-10-06-27---VmlldzozNzE5NTE2?accessToken=8i9nfbgbvu44z24cag0wokh82nxvvx9okpc5uisp5qpxxr0452kdjug93gknn3jz).
    
# Question 8
Approch:
1. Implemented a function to calculate the squared error loss. L = 1/2 * (predicted_value - actual_value) ^ 2
2. Took the outputs of both squared error loss and cross entropy loss for a given set of predictions and actual values.
3. Integrate the outputs of squared error loss and cross entropy loss to see a plot automatically generated on wandb.
4. The inferior performance of Mean Squared Error is attributed to its lack of suitability for probability-based problems. Therefore, for image classification problems, cross-entropy is the more appropriate loss function to use. so cross entropy gives good performance

                  *Results*:                  Mean Square error                Cross Entropy error
                Val Accuracy                    88.87%                                89.53%
                Train Accuracy                  92.51%                                93.21%

The code for implementing these steps can be found in the [link](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/Assignment_1_Question8.ipynb). You can also check the wandb visualisation for question 8 by clicking on the [link](https://api.wandb.ai/links/ed22s009/ybaqty81).

# Question 9
Github [link](https://github.com/swapnilmn/Assignment_1_CS6910)

# Question 10
Approch:
1. With the learning of wandb sweepings for fastion mnist dataset I took 3 best validation accuracy configurations for mnist dataste.
2. From this I got accuracies repectively: 0.9672 ,0.9801,0.9755 for test dataset

code for implementation can be found at [link](https://github.com/swapnilmn/Assignment_1_CS6910/blob/master/Assignment_1_Question10.ipynb). You can also check the wandb visualisation for question 10 by clicking on the [link](https://wandb.ai/ed22s009/Question_10?workspace=user-ed22s009).

# Train.py file
link of [train.py](https://github.com/swapnilmn/Assignment_1-CS6910/blob/master/train_py.py) file 

# Wandb Report
link of [wandb report](https://wandb.ai/ed22s009/Question_4_Best_Model/reports/CS6910-Assignment-1--VmlldzozNzEzNTk1) file
