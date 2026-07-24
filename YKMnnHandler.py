import math
import hashlib
from YKMNueralNet import YathNeuralNet as YNN
import random
import pandas

#Structure for stored configuration
#dev_param_save_frame = [{
    #'id' : int,
    #'peak weights' : list,
    #'peak biases' : list
#}]

def setup():
    Nueral_net = YNN()
    Nueral_net.init_self(2, [32, 16, 8], 1, 0.035)

    training_data = [
        {"inputs": [2.5347, -1.2991], "correct": [0.0]},
        {"inputs": [-1.2794, 0.3771], "correct": [0.0]},
        {"inputs": [-1.4175, -1.7425], "correct": [0.0]},
        {"inputs": [-2.6807, 0.4549], "correct": [0.0]},
        {"inputs": [-2.6496, -2.7367], "correct": [0.0]},
        {"inputs": [-0.8739, -1.9449], "correct": [0.0]},
        {"inputs": [-0.8374, -0.7346], "correct": [0.0]},
        {"inputs": [-0.9497, 1.3911], "correct": [0.0]},
        {"inputs": [2.5998, 4.1696], "correct": [0.0]},
        {"inputs": [0.0295, 5.7019], "correct": [0.0]},
        {"inputs": [-0.2232, -1.8554], "correct": [0.0]},
        {"inputs": [0.7998, -1.3246], "correct": [0.0]},
        {"inputs": [5.1884, 1.8624], "correct": [0.0]},
        {"inputs": [-2.0546, 5.5712], "correct": [0.0]},
        {"inputs": [-0.1494, -0.7444], "correct": [0.0]},
        {"inputs": [2.3768, -0.0828], "correct": [0.0]},
        {"inputs": [-0.745, -3.2434], "correct": [0.0]},
        {"inputs": [3.9609, -0.6018], "correct": [0.0]},
        {"inputs": [0.1823, 0.6337], "correct": [0.0]},
        {"inputs": [-2.6831, -0.3403], "correct": [0.0]},
        {"inputs": [-0.2481, -0.4148], "correct": [0.0]},
        {"inputs": [4.5845, -2.8882], "correct": [0.0]},
        {"inputs": [0.8443, 4.6309], "correct": [0.0]},
        {"inputs": [-5.4044, 1.6356], "correct": [0.0]},
        {"inputs": [-3.8837, 0.9225], "correct": [0.0]},
        {"inputs": [0.5048, -1.9823], "correct": [0.0]},
        {"inputs": [3.7746, 0.5901], "correct": [0.0]},
        {"inputs": [-2.0425, 1.9607], "correct": [0.0]},
        {"inputs": [0.567, 0.6079], "correct": [0.0]},
        {"inputs": [0.0745, 0.6108], "correct": [0.0]},
        {"inputs": [2.6083, 2.3296], "correct": [1.0]},
        {"inputs": [-2.326, -3.8869], "correct": [1.0]},
        {"inputs": [2.3655, 0.357], "correct": [1.0]},
        {"inputs": [-0.3464, 1.6423], "correct": [1.0]},
        {"inputs": [-1.2224, -4.5391], "correct": [1.0]},
        {"inputs": [-0.033, -0.3813], "correct": [1.0]},
        {"inputs": [3.1684, 1.4254], "correct": [1.0]},
        {"inputs": [-4.5584, 3.1811], "correct": [1.0]},
        {"inputs": [-1.6718, -2.869], "correct": [1.0]},
        {"inputs": [1.169, -0.1565], "correct": [1.0]},
        {"inputs": [1.6537, -4.0385], "correct": [1.0]},
        {"inputs": [0.4748, -3.3229], "correct": [1.0]},
        {"inputs": [0.9493, 1.8974], "correct": [1.0]},
        {"inputs": [-0.3805, 4.6194], "correct": [1.0]},
        {"inputs": [-5.1672, -0.6062], "correct": [1.0]},
        {"inputs": [0.9881, 0.4792], "correct": [1.0]},
        {"inputs": [-1.5482, 2.7401], "correct": [1.0]},
        {"inputs": [2.0268, -5.7361], "correct": [1.0]},
        {"inputs": [0.987, 0.0743], "correct": [1.0]},
        {"inputs": [1.3485, 1.4816], "correct": [1.0]},
        {"inputs": [-1.1214, 0.3288], "correct": [1.0]},
        {"inputs": [2.9129, -3.0738], "correct": [1.0]},
        {"inputs": [-1.7051, 4.171], "correct": [1.0]},
        {"inputs": [-3.8805, -0.4293], "correct": [1.0]},
        {"inputs": [5.2665, -1.6195], "correct": [1.0]},
        {"inputs": [-4.9776, -1.5929], "correct": [1.0]},
        {"inputs": [2.1288, -2.0554], "correct": [1.0]},
        {"inputs": [3.4358, -2.0941], "correct": [1.0]},
        {"inputs": [3.892, 3.0683], "correct": [1.0]},
        {"inputs": [1.9716, 1.2377], "correct": [1.0]}
    ]

    validation_data = [
        {"inputs": [1.2541, 2.8912], "correct": [0.0]},
        {"inputs": [-2.1045, -0.4512], "correct": [0.0]},
        {"inputs": [-0.9542, 2.1481], "correct": [0.0]},
        {"inputs": [3.1254, 4.8962], "correct": [0.0]},
        {"inputs": [-1.8421, 4.2104], "correct": [0.0]},
        {"inputs": [0.3541, -0.9214], "correct": [0.0]},
        {"inputs": [4.2105, -1.1478], "correct": [0.0]},
        {"inputs": [-0.5412, -2.1047], "correct": [0.0]},
        {"inputs": [-3.1204, 0.8914], "correct": [0.0]},
        {"inputs": [2.1485, -0.8412], "correct": [0.0]},
        {"inputs": [1.8962, 0.4125], "correct": [1.0]},
        {"inputs": [-1.4512, -3.8962], "correct": [1.0]},
        {"inputs": [2.8412, 1.1045], "correct": [1.0]},
        {"inputs": [-0.2148, 2.8412], "correct": [1.0]},
        {"inputs": [0.8914, 1.2541], "correct": [1.0]},
        {"inputs": [3.8962, 2.1485], "correct": [1.0]},
        {"inputs": [-4.1045, 1.8962], "correct": [1.0]},
        {"inputs": [1.1045, -4.2104], "correct": [1.0]},
        {"inputs": [2.4512, -2.8412], "correct": [1.0]},
        {"inputs": [-2.8912, -1.4512], "correct": [1.0]}
    ]
    # optimization 1 (BCE Early Stopping):
    # using early stopping for network training -> 19th July 2026
    # refer R52, R53
    # note : implementing early stopping to never specify exact number of epochs
    #BCE error for single data point = - ( y(ln(p)) + (1-y)(ln(1-p))
    # but for target = 1, BCE loss function boils down to just -ln(p) -> calculated on rough 52
    # for target = 0, BCE loss function boils down to -ln(1-p) --------------^
    validation_data_losses_temp : int = 0 # setting the temporary variable that is to store the sum of the validation errors (updated/ added upon each datapoint in validation error list
    patience_amt : int = 150 # initializing patience for the validation loss curve
    patience_count : int = 0 # initalizing count for patience
    lowest_val_loss_agg : float # no value initialization of lowest agg val/ lowest point in loss curve
    stop_training_bool : bool = False # initalizing the break training bool
    first_iteration : bool = True # initializing bool to check if first epoch iteration
    peak_weights = [] # initalizing empty peak weights to store the best weights configuration when find lowest point in loss curve
    peak_biases = [] # initalizing empty peak biases to store the best bias configuration when find lowest point in loss curve
    noofepochscheck_dev : int = 0 # initializing dev variable for debugging to see no of epochs used
    # optimization 2 (Learning rate decay):
    decay_rate = 0.994
    while not stop_training_bool: # ensures that it runs another epoch only if it hasnt found the best loss curve point (even after patience)
        initial_lr = Nueral_net.learning_rate # intial learning rate before decay calculation
        noofepochscheck_dev += 1 # incrementing no of epochs used counter
        random.shuffle(training_data)  # shuffle the training data every epoch to make training better
        for data in training_data: # go through every data point once each epoch
            Nueral_net.train(data['inputs'], data['correct']) # training the NN on the inputs and targets provided
        #----------------------------------------------------------------------------------------------------------
        # the NN is now trained for that epoch number (weights and biases modulated)
        #---------------------------------------------------------------------------------------------------------------
        for val_dat in validation_data: # iterating through data in validation set
            val_dat_guess = Nueral_net.feedforward(val_dat['inputs']) # getting the guess(output from NN) after providing each input in validation data 
            if val_dat['correct'][0] == 1.0: # if target was 1, loss/penalty = -ln(guess)
                validation_data_losses_temp += (-((math.log10(val_dat_guess[0][0])) * 2.303)) # adding to the temp storing variable that stores summed values of each agg loss for each validation guess
            else: # if target was 0, loss / penaltly = -ln(1-guess)
                validation_data_losses_temp += (-((math.log10(1 - val_dat_guess[0][0])) * 2.303))

        momentary_agg_loss = validation_data_losses_temp/len(validation_data) # calculating momentary agg loss for that current epoch after all penalties are summed

        if first_iteration: # checking if it is the first epoch
            lowest_val_loss_agg = momentary_agg_loss # setting the lowest possible point in loss curve as the momentary loss agg of first epoch
        else:
            if momentary_agg_loss < lowest_val_loss_agg: # checking to see if it found a new low in the loss curve 
                lowest_val_loss_agg = momentary_agg_loss # setting the lowest point to the newly found lowest point
                peak_weights = [w.copy() for w in Nueral_net.allweightsarray] # copying the weights to store as the best weights
                peak_biases = [b.copy() for b in Nueral_net.biases_array] # copying the biases to store as the best biases
                patience_count = 0 # setting the patience counter back to zero ( since we found a new low)
            else:
                patience_count += 1 # increment the patience counter (since we are either stuck or increasing our loss)

        if patience_count == patience_amt: # wait until the patience is fully tested to the limit
            stop_training_bool = True # stop the training if it has fully reached the limit
        else:
            stop_training_bool = False # let training continue if not
        # optimization : learning rate decay here (originaly linear, now exponential)

        Nueral_net.learning_rate = initial_lr * (decay_rate)
        validation_data_losses_temp = 0 # reset the validation losses sum amount back to zero so it can be setted by the next epoch
        first_iteration = False # set the first epoch bool to false once first epoch is finished

    print(f"Training completed......... \nNumber of epochs used : {str(noofepochscheck_dev)}")
    if peak_weights is not None:
        Nueral_net.allweightsarray = peak_weights
        Nueral_net.biases_array = peak_biases
        print("Succesfully fetched best possible understanding....")
        print("Predicting output for user input...")
    else:
        print("Error69: Could not fetch best possible understanding")
        print("Predicting output for user input (based on last epoch trained understanding *inaccurate*)....")

    #feed forwarding
    X_test = [
        [0.1683, 1.6715],
        [-1.1434, -0.208],
        [-3.5025, -1.5772],
        [-3.729, 4.5263],
        [0.3256, -4.6716],
        [1.5748, 3.0372],
        [1.4128, -2.5684],
        [5.2617, 0.2572],
        [3.2597, -4.5001],
        [-2.6422, 1.0558],
        [-3.3255, 1.8978],
        [1.4956, -0.5464],
        [1.2671, -0.782],
        [-0.7219, 3.2846],
        [-1.1027, 0.8556],
        [0.7525, 3.2056],
        [-2.3077, -1.0749],
        [0.1128, -5.9015],
        [-2.8303, 3.0263],
        [-3.885, -3.1757]
    ]
    for test_input in X_test:
        print(Nueral_net.feedforward(test_input))
    input_to_save(peak_weights=peak_weights, peak_biases=peak_biases) # calling the dev check save function

# SAVE DATA TO CSV FILE
def saving_to_file(dataframe: pandas.DataFrame):
    dataframe.to_csv("SavedConfigurationsNN.csv", index=False, header=False, mode="a")
    print("succesfully saved....")

 # pre set password hash HIDDEN DUE TO SECURITY REASONS
def hashing(passcode:str): # inputed password hashing to match hashes
    inputed_hashed = hashlib.sha256(passcode.encode()).hexdigest()
    return str(inputed_hashed)

def input_to_save(peak_weights:list, peak_biases:list): # get required input for saving
    confirmation = str(input("Would u like to save current knobs ? (y/n) : ")).lower()
    if confirmation == "y":
        input_password = input("Enter Password : ")
        if hashing(str(input_password)) == hashed_pass:
            id_input = input("Enter an id: ")
            try:
                int(id_input)
            except ValueError:
                print("Aborting (Zero Error tolerance)..")
                quit()
            new_frame = {
                'id' : id_input,
                'peak weights' : [peak_weights],
                'peak biases' : [peak_biases]
            }
            print("Saving peak weight and biases parameters ")
            saving_to_file(pandas.DataFrame(new_frame))
        else:
            print("incorrect password: self destructing configurations !")
            quit()
    elif confirmation == "n":
        print("self destructing configurations !")
        quit()
    else:
        print("Aborting (Zero Error tolerance)...")
        quit()
setup() # call entire module to run
