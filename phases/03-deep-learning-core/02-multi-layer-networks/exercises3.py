import math
import random

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        # initialize weights randomly and biases to 0 if None
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]

        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_inputs = inputs
        self.last_output = []

        # every cell is calculated independently in the same layer
        # the calculation follows the rules as defined
        for neuron_idx in range(len(self.weights)):
            z = sum(w * x for w, x in zip(self.weights[neuron_idx], inputs))
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))

        return self.last_output

    def count_params(self):
        n_weights = len(self.weights) * len(self.weights[0])
        n_biases = len(self.biases)

        return n_weights + n_biases

class Network:
    def __init__(self, layers):
        self.layers = layers

    # forward the result from the first layer to the last layer in the network
    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward((current))

        return current

    def count_parameters(self):
        total = 0
        for layer in self.layers:
            total += layer.count_params()

        return total

mnist_net = Network([
    Layer(n_inputs=784, n_neurons=256),
    Layer(n_inputs=256, n_neurons=128),
    Layer(n_inputs=128, n_neurons=10)
])

print(f"The total parameters in 784-256-128-10 network is {mnist_net.count_parameters()}")