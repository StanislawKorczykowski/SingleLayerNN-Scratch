import random
import math


def load_data(file_path):
    data_list = []
    labels_list = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                values = line.strip().split(',')
                data = [float(x) for x in values[:-1]]
                label = values[-1]

                data_list.append(data)
                labels_list.append(label)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return data_list, labels_list


def count_classes(labels_list):
    return set(labels_list)


def random_weights(features_count):
    return [random.random() for _ in range(features_count)]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def train_single_layer_nn(data, labels, num_classes, learning_rate, epochs):
    feature_count = len(data[0])
    class_weights = [random_weights(feature_count) for _ in range(num_classes)]
    class_biases = [random.random() for _ in range(num_classes)]

    for _ in range(epochs):
        for i in range(len(data)):
            x_i = data[i]
            l_i = labels[i]

            for c in range(num_classes):
                z = sum([w * x for w, x in zip(class_weights[c], x_i)]) + class_biases[c]
                output = sigmoid(z)
                target = 1 if l_i == c else 0
                error = target - output

                class_weights[c] = [w + learning_rate * error * x * output * (1 - output) for w, x in
                                    zip(class_weights[c], x_i)]
                class_biases[c] += learning_rate * error * output * (1 - output)

    return class_weights, class_biases


def test_single_layer_nn(data, labels, class_weights, class_biases):
    correct_predictions = 0

    for i in range(len(data)):
        x_i = data[i]
        l_i = labels[i]

        class_outputs = [sigmoid(sum([w * x for w, x in zip(class_weights[c], x_i)]) + class_biases[c]) for c in range(len(class_weights))]
        predicted_label = class_outputs.index(max(class_outputs))

        correct_predictions += 1 if predicted_label == l_i else 0

    return correct_predictions / len(data) * 100


def main():
    train_data, train_labels = load_data("Resources/perceptron.data")
    test_data, test_labels = load_data("Resources/perceptron.test.data")

    label_to_num = {label: i for i, label in enumerate(sorted(set(train_labels)))}

    train_labels = [label_to_num[label] for label in train_labels]
    test_labels = [label_to_num[label] for label in test_labels]

    num_classes = len(label_to_num)

    print("Enter the learning rate:")
    learning_rate = float(input())

    output_weights, output_biases = train_single_layer_nn(train_data, train_labels, num_classes, learning_rate, 100)
    accuracy = test_single_layer_nn(test_data, test_labels, output_weights, output_biases)
    print("Accuracy: " + str(accuracy) + "%")


if __name__ == '__main__':
    main()
