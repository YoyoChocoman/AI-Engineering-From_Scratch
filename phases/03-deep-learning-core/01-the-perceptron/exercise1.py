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

nand_data = [
    ([0, 0], 1),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

print("=== NAND Gate ===")
p_nand = Perceptron(2)
p_nand.train(nand_data)
for inputs, _ in nand_data:
    print(f"  {inputs} -> {p_nand.predict(inputs)}")