from YKMNueralNet import YathNeuralNet as YNN
import random
def setup():
    Nueral_net = YNN()
    Nueral_net.init_self(4, [2], 2, 0.01)

    training_data = [
        {"inputs" : [0.15, 0.23, 0.65, 0.72], "correct" : [1.0, 0.0] },
        {"inputs" :  [0.85, 0.91, 0.12, 0.05], "correct" :  [0.0, 1.0]},
        {"inputs" : [0.45, 0.35, 0.38, 0.41], "correct" :[1.0, 0.0] },
        {"inputs" : [0.72, 0.68, 0.15, 0.22], "correct" : [0.0, 1.0]},
        {"inputs" : [0.10, 0.12, 0.88, 0.95], "correct" : [1.0, 0.0]},
        {"inputs" : [0.90, 0.85, 0.45, 0.30], "correct" : [0.0, 1.0]},
        {"inputs" : [0.33, 0.41, 0.52, 0.48], "correct" : [1.0, 0.0] },
        {"inputs" : [0.60, 0.55, 0.20, 0.15], "correct" : [0.0, 1.0]},
        {"inputs" : [0.05, 0.25, 0.70, 0.60], "correct" : [1.0, 0.0]},
        {"inputs" : [0.80, 0.75, 0.35, 0.40], "correct" : [0.0, 1.0]},
        {"inputs" :  [0.22, 0.31, 0.58, 0.64], "correct" : [1.0, 0.0]},
        {"inputs" : [0.95, 0.92, 0.11, 0.08], "correct" : [0.0, 1.0] },
        {"inputs" : [0.40, 0.38, 0.42, 0.45], "correct" : [1.0, 0.0]},
        {"inputs" : [0.70, 0.71, 0.25, 0.20], "correct" : [0.0, 1.0]},
        {"inputs" : [0.18, 0.14, 0.82, 0.89], "correct" : [1.0, 0.0]},
        {"inputs" : [0.88, 0.82, 0.50, 0.35], "correct" : [0.0, 1.0]},
        {"inputs" : [0.28, 0.36, 0.60, 0.55], "correct" : [1.0, 0.0]},
        {"inputs" : [0.65, 0.62, 0.18, 0.12], "correct" : [0.0, 1.0]},
        {"inputs" : [0.12, 0.20, 0.75, 0.68], "correct" : [1.0, 0.0]},
        {"inputs" : [0.78, 0.81, 0.30, 0.33] , "correct" : [0.0, 1.0]},
        {"inputs" :  [0.500, 0.500, 0.500, 0.500] , "correct" : [0.0, 1.0]},
        {"inputs" :  [0.501, 0.500, 0.500, 0.500] , "correct" : [0.0, 1.0]},
        {"inputs" :  [0.501, 0.500, 0.500, 0.501] , "correct" : [0.0, 1.0]},
        {"inputs" :  [0.499, 0.500, 0.500, 0.500] , "correct" : [1.0, 0.0]},
        {"inputs" :  [0.250, 0.250, 0.250, 0.250], "correct" : [0.0, 1.0]}

    ]
    for i in range(1, 2111100, 1):

        data = random.choice(training_data)
        Nueral_net.train(data['inputs'], data['correct'])


    #feed forwarding
    X_test = [[0.19, 0.22, 0.55, 0.61],  # Class 0 -> (0.19 + 0.22) < (0.55 + 0.61)
        [0.82, 0.88, 0.21, 0.14],
        [0.35, 0.29, 0.40, 0.44],
        [0.67, 0.74, 0.31, 0.28],
        [0.08, 0.17, 0.73, 0.81],
        [0.91, 0.79, 0.38, 0.42],
        [0.42, 0.47, 0.48, 0.51],
        [0.55, 0.63, 0.12, 0.19],
        [0.25, 0.11, 0.66, 0.59],
        [0.71, 0.69, 0.44, 0.39],
        #[0.500, 0.500, 0.501, 0.500], # input is giving wrong data
        [0.500, 0.500, 0.500, 0.500], # (1.000 >= 1.000) -> Expected: Class 1 [0.0, 1.0]
        [0.501, 0.500, 0.500, 0.500] ]
    for test_input in X_test:
        print(Nueral_net.feedforward(test_input))

    #print(Nueral_net.feedforward([0.15,0.25,0.45,0.75]))
    #print(Nueral_net.feedforward([0.85,0.75,0.15,0.30]))
    #print(Nueral_net.feedforward([1,1]))
    #print(Nueral_net.feedforward([1,0]))

    #print(result_out)


setup()
