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
        # print 'input', inputs.shape
        # print 'targets', targets.shape

        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden,
                               inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(
            hidden_inputs)  # signals from hidden layer
        # print "hidden_inputs"
        # print hidden_inputs
        # print "hidden_outputs"
        # print hidden_outputs

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output,
                              hidden_outputs)  # signals into final output layer
        final_outputs = self.activation_function(
            final_inputs)  # signals from final output layer
        # print 'final_inputs'
        # print final_inputs
        # print 'final_output'
        # print final_outputs

        ### Implement the backward pass here ####
        ## Backward pass ###

        # TODO: Output error
        # Output layer error is the difference between desired target and actual output.
        output_errors = targets - final_outputs
        output_grad = self.grid_error(output_errors, final_outputs)
        # print 'output_errors'
        # print output_errors

        # # TODO: Backpropagated error
        # errors propagated to the hidden layer
        # print 'output_error'
        # print output_grad
        # print 'weights_hidden_output'
        # print self.weights_hidden_to_output
        hidden_errors = np.dot(output_grad, self.weights_hidden_to_output)
        # print 'hidden_errors'
        # print hidden_errors
        # print 'hidden_activations'
        # print hidden_outputs
        # hidden layer gradients
        hidden_grad = self.grid_error(hidden_errors, hidden_outputs.T)
        # error * output * (1 - output)
        # print'hidden_error'
        # print hidden_grad

        # # TODO: Update the weights
        # print "output_grad"
        # print output_grad
        # print "hidden_outputs"
        # print hidden_outputs
        # print 'self.weights_hidden_to_output'
        # print self.weights_hidden_to_output
        self.weights_hidden_to_output += self.delta_w(self.lr, output_grad,
                                                      hidden_outputs).T  # update hidden-to-output weights with gradient descent step
        # print 'self.weights_input_to_hidden'
        # print self.weights_input_to_hidden
        # print 'hidden_grad'
        # print hidden_grad
        # print 'inputs'
        # print inputs
        self.weights_input_to_hidden += self.delta_w(self.lr, hidden_grad,
                                                     inputs).T
        # output_error
        # [[-0.24889032]]
        # weights_hidden_output
        # [[0.73869277 - 0.65114888]]
        # hidden_errors
        # [[-0.18385348  0.16206466]]
        # hidden_activations
        # [[0.88475718]
        #  [0.17508056]]
        # hidden_grad
        # [[-0.01874605  0.01652442]
        #  [-0.02655347  0.02340657]]

        # update input-to-hidden weights with gradient descent step

    # he = np.array([[-0.18385348, 0.16206466]])
    # a = 1
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden,
                               inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(
            hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output,
                              hidden_outputs)  # signals into final output layer
        final_outputs = self.activation_function(
            final_inputs)  # signals from final output layer

        return final_outputs

    def grid_error(self, error, output):
        output_error = error * output * (1 - output)
        return output_error

    def delta_w(self, lr, grid_error, input_before_weight):
        return lr * input_before_weight * grid_error


