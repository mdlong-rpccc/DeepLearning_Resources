# Resources for understanding Machine Learning Techniques
This is a collection of useful material for learning statistical approaches

## Survival Analysis & Deep Learning
https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/

## Regularization in Deep Learning
https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/

## Backpropagation
Backpropagation is an algorithm for training artificial neural networks, especially multi-layer perceptrons (MLPs). It is a method for updating the weights of the network so as to minimize a loss function that measures the difference between the predicted outputs of the network and the true outputs.

Backpropagation is a supervised learning algorithm, meaning that it requires labeled training data in order to learn the correct weights of the network. During the training process, the input data is passed forward through the network, producing an output. The loss is then computed between the predicted output and the true output, and this loss is used to compute the gradient of the loss with respect to the weights of the network.

The gradient information is then used to update the weights of the network in a direction that will reduce the loss. This process is repeated multiple times, with each iteration leading to a reduction in the loss. The weights are updated using gradient descent, which means that the update rule for the weights is proportional to the negative gradient of the loss with respect to the weights.

Backpropagation is an efficient and effective method for training neural networks, and it has been the foundation of many advances in the field of deep learning. It is also the basis for many variations and extensions of the algorithm, such as momentum-based optimization, Adagrad, Adam, and others.


## Types of Artificial Neural Networks

### Introduction to ANN
https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/

A neural network is a type of machine learning model inspired by the structure and function of the human brain. The basic building block of a neural network is the artificial neuron, which takes inputs, performs a computation on them, and then produces an output.

Multiple artificial neurons are connected together to form a network, with the output of one neuron serving as the input to another. The connections between neurons are associated with weights that are learned during training. The training process involves adjusting these weights so that the neural network can accurately map inputs to outputs for a given set of examples.

Neural networks can be used for a variety of tasks, such as image classification, language translation, and even playing games. They are particularly powerful for tasks that involve finding patterns in complex, high-dimensional data.

There are many different types of neural networks, each with its own strengths and weaknesses. Some popular types include feedforward networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs), and long short-term memory networks (LSTMs). The choice of which type of network to use depends on the specifics of the task and the data.


### Graph Neural Networks
https://distill.pub/2021/gnn-intro/

A graph neural network (GNN) is a type of neural network designed to operate on graph-structured data, such as social networks, molecular structures, or transportation networks. In a graph, nodes represent entities and edges represent relationships or connections between them.

In a GNN, each node in the graph is associated with a vector of features, and the goal is to learn a function that can produce a representation for each node that incorporates information from both its own features and the features of its neighbors in the graph. This representation can then be used for various tasks, such as node classification, graph classification, or link prediction.

GNNs use a message-passing mechanism to propagate information from the neighbors of a node to the node itself. The message-passing mechanism is typically implemented as a neural network layer that takes as input the features of a node and the features of its neighbors, and produces a new representation for the node. This process is repeated multiple times to allow the information to flow deeper into the graph.

GNNs have been shown to be effective for a wide range of graph-based problems, and have been used in areas such as chemistry, physics, social sciences, and recommendation systems. Some popular types of GNNs include graph convolutional networks (GCNs), graph attention networks (GATs), and graph auto-encoders (GAEs).


### Sparse Neural Networks
A sparse neural network is a type of neural network that has been designed to have fewer parameters or connections, compared to a dense neural network of the same size. This sparsity can be achieved in various ways, such as pruning connections that are found to be less important, or using structured sparsity patterns in the network architecture.

The main motivation for using sparse neural networks is to reduce the computational cost and memory footprint of the model, which can be especially important for large-scale models or when deploying models on resource-constrained devices. Sparsity can also make the model more interpretable and easier to analyze, as well as lead to better generalization performance.

There are various techniques for inducing sparsity in neural networks, including weight pruning, structured sparsity, and low-rank factorization. In weight pruning, individual connections are removed based on their importance, as determined by a metric such as the magnitude of the weight. In structured sparsity, the sparsity pattern is imposed on the network in a structured way, such as using a specific activation function or applying a constraint on the network weights. In low-rank factorization, the weight matrix is decomposed into the product of two lower-dimensional matrices, leading to a compact representation of the network that still retains most of the important information.

Sparse neural networks can be trained using standard optimization algorithms, with the added constraint of inducing sparsity. There are also specialized algorithms that have been developed specifically for training sparse neural networks, such as the alternating direction method of multipliers (ADMM) and the proximal gradient descent method.


### Convolutional Neural Networks
A convolutional neural network (ConvNet or CNN) is a type of neural network that is specifically designed to work with image data. ConvNets are inspired by the structure of the visual cortex in the human brain and are used for tasks such as image classification, object detection, and semantic segmentation.

ConvNets are constructed using a series of convolutional layers, each of which performs a convolution operation on the input data. A convolution operation involves sliding a small filter (also known as a kernel or feature detector) over the input data and computing the dot product between the values in the filter and the input data. This operation is repeated many times with different filters to capture different features of the input data. The outputs of the convolutional layers are then fed into a series of activation functions, which introduce non-linearity into the model.

In addition to convolutional layers, ConvNets also include pooling layers, which are used to reduce the spatial dimensions of the data and to provide some translation invariance. Pooling layers are typically implemented as max-pooling layers, which keep the maximum value of each group of input values.

The outputs of the pooling layers are then passed through a series of fully-connected (dense) layers, which are used to make predictions based on the learned features. The final layer in a ConvNet is typically a softmax layer, which outputs a probability distribution over the possible classes.

ConvNets are trained using backpropagation and gradient descent, with the goal of minimizing the loss function between the predicted probabilities and the ground truth labels. ConvNets have been shown to be very effective for image classification tasks and are widely used in a variety of computer vision applications.

