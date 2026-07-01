# Step 1: The Perceptron class
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        # initializes all weights and bias top 0.0
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        # total = bias + the sum of all weights * inputs
        # prediction = bool(total >= 0)
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >=0 else 0

    def train(self, training_data, epochs=100):
        # epoch is the training rounds
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                # make predictions and find out the error
                prediction = self.predict(inputs)
                error = target - prediction

                # adjust the weights if there's error
                # learning formula = old_weight + (learning_rate * error * input)
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]

                    self.bias += self.lr * error

            if errors == 0:
                print(f"Train Complete! It reaches full points in {epoch + 1} round.")
                return

        print(f"Train Fail! (after {epochs} rounds)")

# Step 2: Train on logic gates
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")

# Step 3: Watch XOR fail
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "Wrong"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")

# Step 4: Solve XOR with two layers
# the logic of xor = (A OR B) AND (A NAND B)
def xor_network(x1, x2):
    # 1. the first hidden cell (A OR B)
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    # 2. the second hidden cell (A NAND B)
    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    # 3. the output cell (first AND second)
    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = 1.5

    # start forwarding
    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])

    return output

print(f"\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")

# Step 5: Train a Two-Layer Network
import random
import math

class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        # randomly guesses the weights and bias with two hidden layer
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]

        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)

        self.lr = learning_rate

    def sigmoid(self, x):
        # a new judgement with a smooth curve between 0 ~ 1
        x = max(-500, min(500, x))  # preventing overflowing caused by enormous numbers
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        # calculates all along the way
        self.inputs = inputs
        self.hidden_outputs = []

        # the result of the two hidden cells
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))

        # the results of the final output
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)

        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            for inputs, target in training_data:
                # first calculate it once and find out the error
                output = self.forward(inputs)
                error = target - output

                # output layer's fault (error * sigmoid's gradient)
                d_output = error * output * (1 - output)

                # distributes the fault to the hidden layer by weights
                saved_w_ouput = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_ouput[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                # update output layer's weight
                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                # update hidden layer's weight
                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]

print("\n=== Start Training with TwoLayerNetwork ===")
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")