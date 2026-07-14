# Next steps : upgrade this architecture to support multiple hidden layers (check IRL papers for plan)

import numpy as np


#import math

class YathNeuralNet:
    rng = np.random.default_rng()
    def init_self(self, input_nuerons, hidden_layer_array, output_nuerons, learning_rate):
        #initializing all nuerons based on caller requirement
        self.input_nuerons = input_nuerons
        self.output_nuerons = output_nuerons
        #initialzing nueron amount array
        self.hidden_nueron_array = hidden_layer_array
        #initalize temporary hidden matrices array
        self.hidden_matrix_PL_array : list[np.ndarray | None] = [None] * (len(self.hidden_nueron_array))
        #initializing Error Array
        self.allErrors : list[np.ndarray | None] = [None] * (len(self.hidden_nueron_array) + 1)
        #initializing learning rate
        self.learning_rate = learning_rate
        # initialzing temporary weight matrices and weight gradients array and biases and biases gradient
        self.allweightsarray : list[np.ndarray | None] = [None] * (len(self.hidden_nueron_array) + 1)
        self.allweightgradeintsarray : list[np.ndarray | None] = [None] * (len(self.allweightsarray))
        self.biases_array : list[np.ndarray | None] = [None] * (len(self.allweightsarray))
        self.biases_gradients_array : list[np.ndarray | None] = [None] * (len(self.allweightsarray))

        #weight and biases initialization loop -> loops over only length initialized all weights array
        for i in range(0, len(self.allweightsarray)):
            if i == 0: # for first weight btw Input layer and Foremost Hidden Layer
                self.allweightsarray[i] = self.rng.normal(0, 1.0, (self.hidden_nueron_array[i], self.input_nuerons))
                self.biases_array[i] = self.rng.normal(-1.0, 1.0, (self.hidden_nueron_array[i], 1))
            elif i == len(self.allweightsarray) - 1: # for last weight btw Outermost Hidden Layer and Output layer
                self.allweightsarray[i] = self.rng.normal(0, 1.0, (self.output_nuerons, self.hidden_nueron_array[i - 1]))
                self.biases_array[i] = self.rng.normal(-1.0, 1.0, (self.output_nuerons, 1))
            else: # for every other weight btw each hidden layer
                self.allweightsarray[i] = self.rng.normal(0, 1.0, (self.hidden_nueron_array[i], self.hidden_nueron_array[i-1]))
                self.biases_array[i] = self.rng.normal(-1.0, 1.0, (self.hidden_nueron_array[i], 1))

    def sigmoid(self,x):  # the sigmoid function for activation
        return 1 / (1 + np.exp(-x))  # clean probablity between 0 and 1

    def ultasigmoid(self, x): # ultasigmoid or the derivative of the sigmoid function
        return x * (1 - x)

    def feedforward(self, input_array):
        input_matrix = np.array(input_array).reshape(-1, 1) # this makes the list into a matrix and converts it from 1x2 to 2x1 for our matrix math
        # loop to calculate values of each elemement in every nueron of hidden matrices in every layer
        for x in range(0, len(self.hidden_matrix_PL_array)):
            if x == 0: # for foremost Hidden Layer (with Input)
                self.hidden_matrix_PL_array[x] = self.sigmoid(np.dot(self.allweightsarray[x], input_matrix) + self.biases_array[x])
            else: # for every other hidden layer, only goes until the last index in no of layers
                self.hidden_matrix_PL_array[x] = self.sigmoid(np.dot(self.allweightsarray[x], self.hidden_matrix_PL_array[x - 1]) + self.biases_array[x])
        # calculates the output matrix result
        output_matrix = self.sigmoid(np.dot(self.allweightsarray[-1], self.hidden_matrix_PL_array[-1]) + self.biases_array[-1])
        # converts into readable output
        converted_list = output_matrix.tolist()
        # rounding all values
        for i in range(0, len(converted_list)):
            converted_list[i] = round(converted_list[i][0])

        return converted_list

    # Backpropogation : transpose the weight matrix HO, mulitply scalar Error (previous)
    # weight gradient = (lr * Error_mat * (act_g(1 - act_g))) x Transpose_layer (O -> H, H -> I)
    def train(self, input_array, target_array):
        #converting user given arrays to matrices for further calculation
        input_matrix = np.array(input_array).reshape(-1, 1)
        target_matrix = np.array(target_array).reshape(-1, 1)

        #hidden calculation - > caculating hidden matrices per layer
        for x in range(0, len(self.hidden_matrix_PL_array)):
            if x == 0:  # for foremost Hidden Layer (with Input)
                self.hidden_matrix_PL_array[x] = self.sigmoid(np.dot(self.allweightsarray[x], input_matrix) + self.biases_array[x])
            else: # for every other hidden layer
                self.hidden_matrix_PL_array[x] = self.sigmoid(np.dot(self.allweightsarray[x], self.hidden_matrix_PL_array[x - 1]) + self.biases_array[x])

        #output_calculation - > calculating the final output matrix layer
        output_matrix = self.sigmoid(np.dot(self.allweightsarray[-1], self.hidden_matrix_PL_array[-1]) + self.biases_array[-1])## calculating output matrix


        #Backpropogation loop -> stores all errors for each hidden layer including output layer in dedicated errors array
        for y in range(len(self.allErrors)-1, -1, -1): # loops backwards
            if y == len(self.allErrors)-1: # for output matrix error
                self.allErrors[y] = target_matrix - output_matrix
            else: # for all other hidden layers
                self.allErrors[y] = np.dot(np.transpose(self.allweightsarray[y+1]), self.allErrors[y+1])


        # Gradient Calculation
        # Calculating Weight gradient and biases gradient for weights between Hidden and Output layers and biases of each hidden layer and output layer
        #for z in range(0, len(self.allweightgradeintsarray)):
            #if z == 0:
                #self.allweightgradeintsarray[z] = np.dot(np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(self.hidden_matrix_PL_array[z])), np.transpose(input_matrix))
                #self.biases_gradients_array[z] = np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(self.hidden_matrix_PL_array[z]))
            #elif z == len(self.allweightgradeintsarray) -1:
                #self.allweightgradeintsarray[z] = np.dot(np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(output_matrix)), np.transpose(self.hidden_matrix_PL_array[z-1]))
                #self.biases_gradients_array[z] = np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(output_matrix))
            #else:
                #self.allweightgradeintsarray[z] = np.dot(np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(self.hidden_matrix_PL_array[z])),np.transpose(self.hidden_matrix_PL_array[z - 1]))
                #self.biases_gradients_array[z] = np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(self.hidden_matrix_PL_array[z]))

        for z in range(0, len(self.allweightsarray)):
            if z == 0:
                self.allweightsarray[z] += np.dot(np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(self.hidden_matrix_PL_array[z])), np.transpose(input_matrix))
                self.biases_array[z] += np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(self.hidden_matrix_PL_array[z]))
            elif z == len(self.allweightgradeintsarray) -1:
                self.allweightsarray[z] += np.dot(np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(output_matrix)), np.transpose(self.hidden_matrix_PL_array[z-1]))
                self.biases_array[z] += np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(output_matrix))
            else:
                self.allweightsarray[z] += np.dot(np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(self.hidden_matrix_PL_array[z])),np.transpose(self.hidden_matrix_PL_array[z - 1]))
                self.biases_array[z] += np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ultasigmoid(self.hidden_matrix_PL_array[z]))

        # tuning the weights and biases
        #for tempid in range(0, len(self.allweightsarray)):
            #self.allweightsarray[tempid] += self.allweightgradeintsarray[tempid]
            #self.biases_array[tempid] += self.biases_gradients_array[tempid]