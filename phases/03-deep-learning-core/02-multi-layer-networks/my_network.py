import math
import random

# Step 1: Sigmoid Activation
# same as the previous sigmoid
def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

# Step 2: Layer Class
# n_inputs = last layer's neuron number
# n_neurons = this layer's neuron number
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

# Step 3: Network Class
class Network:
    def __init__(self, layers):
        self.layers = layers

    # forward the result from the first layer to the last layer in the network
    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward((current))

        return current

# Step 4: XOR with Hand-Tuned Weights
hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[[20.0, 20.0], [-20.0, -20.0]],
    biases=[-10.0, 30.0],
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[[20.0, 20.0]],
    biases=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("=== Muti-Layer-Networks (XOR) ===")

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected})")

# Step 5: Circle Classification
random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

# with random weights
circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1),
])

correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

print(f"\nAccuracy with random weights: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")