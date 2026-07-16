from YKMNueralNet import YathNeuralNet as YNN
import random
def setup():
    Nueral_net = YNN()
    Nueral_net.init_self(2, [16, 16, 8], 1, 0.001)

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


    # using shuffle training with epochs (i)
    for i in range(0, 3000):
        random.shuffle(training_data)  # shuffle the training data every epoch to make training more better
        for data in training_data: # go through every data point once each epoch
            Nueral_net.train(data['inputs'], data['correct'])


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

    #print(Nueral_net.feedforward([0.15,0.25,0.45,0.75]))
    #print(Nueral_net.feedforward([0.85,0.75,0.15,0.30]))
    #print(Nueral_net.feedforward([1,1]))
    #print(Nueral_net.feedforward([1,0]))

    #print(result_out)


setup()
