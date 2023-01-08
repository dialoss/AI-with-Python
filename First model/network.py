import numpy as np
import scipy.special as scp
import matplotlib.pyplot as plt

class nueralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        self.activation_function = lambda x: scp.expit(x)

    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot(output_errors * final_outputs * (1.0 - final_outputs), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(inputs))

def test():
    f = open("my_test.csv", 'r')
    test_data = f.readlines()
    f.close()

    correct = 0
    incorrect = 0

    for record in test_data:
        values = record.split(',')
        correct_label = int(values[0])
        inputs = np.asfarray(values[1:]) / 255 * 0.99 + 0.01
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        if label == correct_label:
            correct += 1
        else:
            incorrect += 1
    print(correct / (correct + incorrect) * 100, '%')

i_nodes = 784
h_nodes = 100
o_nodes = 10
l_rate = 0.3
epochs = 2


n = nueralNetwork(i_nodes, h_nodes, o_nodes, l_rate)

def train():
    global n, epochs
    f = open("my_train.csv", 'r')
    train_data = f.readlines()
    f.close()
    for epoch in range(epochs):
        for record in train_data:
            values = record.split(',')

            scaled_input = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01

            img = scaled_input.reshape((28, 28))
            plt.imshow(img)
            plt.show()
            targets = np.zeros(o_nodes) + 0.01
            targets[int(values[0])] = 0.99

            n.train(scaled_input, targets)


def save_model():
    f = open("model.txt", 'w')
    for i in n.wih:
        for j in i:
            f.write(str(j) + ' ')
        f.write('\n')
    f.write('next\n')
    for i in n.who:
        for j in i:
            f.write(str(j) + ' ')
        f.write('\n')
    f.close()

def load_model():
    global n
    f = open("model.txt", 'r')
    for i in range(n.wih.shape[0]):
        value_str = f.readline().split()
        for j in range(len(value_str)):
            n.wih[i][j] = float(value_str[j])
    a = f.readline()
    for i in range(n.who.shape[0]):
        value_str = f.readline().split()
        for j in range(len(value_str)):
            n.who[i][j] = float(value_str[j])
    f.close()

def query(input):
    global n
    values = input
    inputs = np.asfarray(values[:]) / 255 * 0.99 + 0.01
    outputs = n.query(inputs)
    return outputs


# train()
# test()
# save_model()
# load_model()
# test()
