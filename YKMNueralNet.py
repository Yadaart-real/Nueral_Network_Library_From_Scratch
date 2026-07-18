
# Optimization and betterment : right now code works for relu activaiton because relu function derivative gives zero even for a negative value so it doesn't matter if we pass in pre activation values or post activation values of matrix since either will give same output as again relu derviative gives 0 for negative values and 0 for 0 aswell
# In the future change the gradient descent code to include my new calculation having derivative taken of pre activation layer to allow for more better activation functions
# another issue ive run into is DEADRLU's so during original weight initialization im hard coding a standard deviation of 0.1 and normal distribution of 0, almost 50% of nuerons are deadRLU's which means they get negative values and remain zero thus never changing and thus not affecting the weights during training
# solution to this issue: He initalization of weights ( a design to keep ReLU's alive) -> scales the weights based on the number of incoming connections  basically if we take the weight between Input layer and first hidden layer, we get the number of incoming connections as the no. of nuerons of input layer, since each nueron connects to the hidden layers nueron atleast once
# HE INITIALIZATION -> Specifically for RELu breaks for other activations
# LeakyRelu's implemented to further stop dead Relu's during training aswell
# LIBRARY GOT stable SCORE OF 18/20 FOR NOW improved from previous 16/20 : Due to BSE implementation

# Logs :
# Optimization: 1) successfully stored pre activated values for hidden layers and output layers, for other activation derivatives to work
                    # Note : for sigmoid and leaky relu, storing specific preactivated values is not neccessary as input for sigmoid derivative is the sigmoid activated value itself, and the input for leaky relu derivative relies more on the sign/positive negative or zero,  more than the exact value, and thus can be inferred from the post activated values itself

import numpy as np

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
        self.hidden_matrix_PL_array_preactivation : list[np.ndarray | None] = [None] * (len(self.hidden_nueron_array))
        #initializing Error Array
        self.allErrors : list[np.ndarray | None] = [None] * (len(self.hidden_nueron_array) + 1)
        #initializing learning rate
        self.learning_rate = learning_rate
        #intializing LeakyRelU alpha
        self.leaky_relu_alpha : float = 0.01
        # initialzing temporary weight matrices and biases matrices arrays
        self.allweightsarray : list[np.ndarray | None] = [None] * (len(self.hidden_nueron_array) + 1)

        self.biases_array : list[np.ndarray | None] = [None] * (len(self.allweightsarray))


        #weight and biases initialization loop -> loops over only length initialized all weights array
        #recent HE initalization implementation
        for i in range(0, len(self.allweightsarray)):
            # recently implementing He initi std dev = root 2/ n(incoming)

            if i == 0: # for first weight btw Input layer and Foremost Hidden Layer
                std_dev = np.sqrt(2.0/ self.input_nuerons)
                self.allweightsarray[i] = self.rng.normal(0, std_dev, (self.hidden_nueron_array[i], self.input_nuerons))
                self.biases_array[i] = np.ones((self.hidden_nueron_array[i], 1)) * 0.01
            elif i == len(self.allweightsarray) - 1: # for last weight btw Outermost Hidden Layer and Output layer
                std_dev = np.sqrt(1.0 / self.hidden_nueron_array[-1])
                self.allweightsarray[i] = self.rng.normal(0, std_dev, (self.output_nuerons, self.hidden_nueron_array[-1]))
                self.biases_array[i] = np.zeros((self.output_nuerons, 1))
            else: # for every other weight btw each hidden layer
                std_dev = np.sqrt(2.0 / self.hidden_nueron_array[i-1])
                self.allweightsarray[i] = self.rng.normal(0, std_dev, (self.hidden_nueron_array[i], self.hidden_nueron_array[i-1]))
                self.biases_array[i] = np.ones((self.hidden_nueron_array[i], 1)) * 0.01

    # The activation graveyard -------
    def relu_act(self, x): # the relu function for activation of hidden layers, processes the values to be only positive
        return np.maximum(0, x)

    def ulta_relu_act(self, x): # the derivative of the relu function at x = 0 is undefined, but we keep it 0
        return np.where(x <= 0, 0, 1)  # returns 1 if x > 0 and 0 if x < or = to 0
    #---------------------------------------
    #The land of activation survivors
    def sigmoid(self, x): # the sigmoid function for activation of output layers
        x = np.clip(x, -15, 15)
        return 1 / (1 + np.exp(-x))  # clean probablity between 0 and 1

    def ultasigmoid(self, x):  # ultasigmoid or the derivative of the sigmoid function
        return x * (1 - x)

    def leaky_relu_act(self, x): # the leaky relu activation function, does not allow gradient value to become zero effectively keeping a tiny amount hence preventing stagnant non learning weights, and thus preventing dead gradients/Relu's
        return np.where(x > 0, x, self.leaky_relu_alpha * x)

    def ulta_leaky_relu_act(self, x): # keeps the alpha (small value) preventing gradient death by zero multiplication, as opposed to normal relu
        return np.where(x > 0, 1, self.leaky_relu_alpha)

    # Feed Forward Algorithm
    def feedforward(self, input_array):
        input_matrix = np.array(input_array).reshape(-1, 1) # this makes the list into a matrix and converts it from 1x2 to 2x1 for our matrix math
        # loop to calculate values of each elemement in every nueron of hidden matrices in every layer
        for x in range(0, len(self.hidden_matrix_PL_array)):
            if x == 0: # for foremost Hidden Layer (with Input)
                self.hidden_matrix_PL_array[x] = self.leaky_relu_act(np.dot(self.allweightsarray[x], input_matrix) + self.biases_array[x])
            else: # for every other hidden layer, only goes until the last index in no of layers
                self.hidden_matrix_PL_array[x] = self.leaky_relu_act(np.dot(self.allweightsarray[x], self.hidden_matrix_PL_array[x - 1]) + self.biases_array[x])
        # calculates the output matrix result
        output_matrix = self.sigmoid(np.dot(self.allweightsarray[-1], self.hidden_matrix_PL_array[-1]) + self.biases_array[-1])
        # converts into readable output
        converted_list = output_matrix.tolist()
        # rounding all values
        #for i in range(0, len(converted_list)):
            #converted_list[i] = round(converted_list[i][0])

        return converted_list


    # weight gradient = (lr * Error_mat * (act_g(1 - act_g))) x Transpose_layer (O -> H, H -> I)
    def train(self, input_array, target_array):
        #converting user given arrays to matrices for further calculation
        input_matrix = np.array(input_array).reshape(-1, 1)
        target_matrix = np.array(target_array).reshape(-1, 1)

        #hidden calculation - > caculating hidden matrices per layer
        for x in range(0, len(self.hidden_matrix_PL_array)):
            if x == 0:  # for foremost Hidden Layer (with Input)
                self.hidden_matrix_PL_array_preactivation[x] = np.dot(self.allweightsarray[x], input_matrix) + self.biases_array[x] # raw calculation of hidden layer value storing preactivated values for derivative of activation function
                self.hidden_matrix_PL_array[x] = self.leaky_relu_act(self.hidden_matrix_PL_array_preactivation[x]) # passing through activation function
            else: # for every other hidden layer
                self.hidden_matrix_PL_array_preactivation[x] = np.dot(self.allweightsarray[x], self.hidden_matrix_PL_array[x-1]) + self.biases_array[x]
                self.hidden_matrix_PL_array[x] = self.leaky_relu_act(self.hidden_matrix_PL_array_preactivation[x])

        #output_calculation - > calculating the final output matrix layer
        output_matrix_pre_activ = np.dot(self.allweightsarray[-1], self.hidden_matrix_PL_array[-1]) + self.biases_array[-1] ## calculating output matrix and storing preactivation (although not neccessary for sigmoid function)
        output_matrix = self.sigmoid(output_matrix_pre_activ) # passing through activation function

        # Backpropogation : transpose the weight matrix HO
        #Backpropogation loop -> stores all errors for each hidden layer including output layer in dedicated errors array
        for y in range(len(self.allErrors)-1, -1, -1): # loops backwards
            if y == len(self.allErrors)-1: # for output matrix error
                #self.allErrors[y] = np.multiply((target_matrix - output_matrix), self.ultasigmoid(output_matrix)) # use this if I want to use MSE as in MSE the sigmoid derivative doesnt cancel and must also be baked for every hidden weight gradient calculation aswell along with the respective activation layer derivative calculated seperately in gradient function below
                self.allErrors[y] = target_matrix - output_matrix # use this if I want to use BSE since in BSE the sigmoid derivative cancels out with the derviative of the BSE loss for each layer and so there is no need to pass it into the raw activated errors in the first place as it gets cancelled out along with the denominator of derivative of BSE loss function

            else:
                self.allErrors[y] = np.dot(np.transpose(self.allweightsarray[y+1]), self.allErrors[y+1])


        # Gradient Calculation, # weight gradient = (lr * Error_mat(ahead layer) * derivative w.r.t act function of ahead layer x Transpose_layer (O -> H, H -> I)
        # Calculating Weight gradient and biases gradient for weights between Hidden and Output layers and biases of each hidden layer and output layer
        # Tuning is done simultaneously
        # note : originally using MSE loss function for entire network (keep the sigmoid derivative hadamard product at outermost weight gradient calculation) , if want to change to BSE loss function of better binary classification (unintentionally i wrote such beautifull code that the raw errors passed down through back propogation
            # are basically the same since in BSE the sigmoid derivative cancels out leaving only the raw error being passed back which i already do and hence the sigmoid derivative is no longer needed as it again just cancels with the gradient of BSE loss function, and the hidden layer errors never get that bad denominator of guess(1 - guess) as it is
            # completely eradicated with the first output layer error itself(upon multiuplying by sigmoid derivative) and hence never passed down, and so since i am already applying the sigmoid derivative in the now no longer raw errors (though they are the same as before),
            # I can just drop the sigmoid derivative when calculating the weight gradient for the outermost weight, but keeping the activation derivative for weights between hidden layers(themselves) and input layer, since the error passed into the gradient calculation for said errors already has inbuilt sigmoid in it during the backpropogation previously explained,
            # we must keep only the derivative for the specific activation funciton of the hidden layer due to chain rule
            # i must admit i was doing MSE wrong i completely forgot the backpropogated error roots from the error calculated on the simgoid activated output hence the sigmoid activated values go backwards and hence we need the derivative of the sigmoid activation and the respective activaiton funciton derivative for that hidden layer due to chain rule and so in MSE the sigmoid for the first error doesnt actually cancel out and still remains in the hidden layer calculations aswell as it should
        for z in range(0, len(self.allweightsarray)):
            if z == 0:
                gradient_weight_temp = np.dot(np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ulta_leaky_relu_act(self.hidden_matrix_PL_array[z])), np.transpose(input_matrix))
                self.allweightsarray[z] += np.clip(gradient_weight_temp, -1.0, 1.0) # Gradient clipping, (to stop those weights from getting as thicc as khalifas ass and exploding) cus the weights compound and increase
                self.biases_array[z] += np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ulta_leaky_relu_act(self.hidden_matrix_PL_array[z]))
            elif z == len(self.allweightsarray) -1:
                gradient_weight_temp = np.dot(np.multiply(self.learning_rate, self.allErrors[z]), np.transpose(self.hidden_matrix_PL_array[z-1])) # drop the sigmoid derivative as we bake it into error
                self.allweightsarray[z] += np.clip(gradient_weight_temp, -1.0, 1.0) # Gradient Clipping
                self.biases_array[z] += np.multiply(self.learning_rate, self.allErrors[z])
            else:
                gradient_weight_temp = np.dot(np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ulta_leaky_relu_act(self.hidden_matrix_PL_array[z])),np.transpose(self.hidden_matrix_PL_array[z - 1]))
                self.allweightsarray[z] += np.clip(gradient_weight_temp, -1.0, 1.0) # gradient clipping
                self.biases_array[z] += np.multiply(np.multiply(self.learning_rate, self.allErrors[z]), self.ulta_leaky_relu_act(self.hidden_matrix_PL_array[z]))
