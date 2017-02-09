# Implement the sigmoid function to use as the activation function.
# Set self.activation_function in __init__ to your sigmoid function.

# Implement the forward pass in the train method.

# Implement the backpropagation algorithm in the train method,
# including calculating the output error.

# Implement the forward pass in the run method.

import numpy as np
import random


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        random.seed(1)
        self.weights_input_to_hidden = np.random.normal(0.0,
                                                        self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes,
                                                         self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0,
                                                         self.output_nodes ** -0.5,
                                                         (self.output_nodes,
                                                          self.hidden_nodes))
        self.lr = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        print 'input', inputs.shape
        print 'targets', targets.shape

        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden,
                               inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(
            hidden_inputs)  # signals from hidden layer
        print "hidden_inputs"
        print hidden_inputs
        print "hidden_outputs"
        print hidden_outputs

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output,
                              hidden_outputs)  # signals into final output layer
        final_outputs = self.activation_function(
            final_inputs)  # signals from final output layer
        print 'final_inputs'
        print final_inputs
        print 'final_output'
        print final_outputs

        ### Implement the backward pass here ####
        ## Backward pass ###

        # TODO: Output error
        # Output layer error is the difference between desired target and actual output.
        output_errors = targets - final_outputs
        output_grid = self.grid_error(output_errors, final_outputs)
        print 'output_errors'
        print output_errors

        # # TODO: Backpropagated error
        hidden_errors = # errors propagated to the hidden layer
        hidden_grad = # hidden layer gradients

        # # TODO: Update the weights
        self.weights_hidden_to_output += self.delta_w(self.lr, output_errors_grid, hidden_outputs) # update hidden-to-output weights with gradient descent step
        # self.weights_input_to_hidden += # update input-to-hidden weights with gradient descent step

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        """

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = # signals into hidden layer
        hidden_outputs = # signals from hidden layer

        # TODO: Output layer
        final_inputs = # signals into final output layer
        final_outputs = # signals from final output layer

        return final_outputs
        """

    def grid_error(self, error, output):
        output_error = error * output * (1 - output)
        return output_error

    def delta_w(self, lr, grid_error, input_before_weight):
        return lr * grid_error * input_before_weight



def MSE(y, Y):
    return np.mean((y - Y) ** 2)
