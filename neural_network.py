import numpy as np, matplotlib.pyplot as plt
import copy, math


class Point:
    def __init__(self, values, label):
        self.values = values # values for each point
        self.label = label # label that shows each point belongs to which cluster
        self.predicted_label = None # label that is predicted for test data


# sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# running the gradient descent algorithm for single-layer network
def gradient_descent_linear(train_data, learning_rate=0.01, epoch=5000):
    m = len(train_data)
    x_arr = [np.append(np.array(1), point.values) for point in train_data]
    y_arr = [np.array(point.label) for point in train_data]
    w = np.append(np.array([0]), np.random.normal(loc=0.0, scale=0.1, size=len(x_arr[0]) - 1))
    costs = []
    for i in range(epoch):
        grad = np.zeros(len(w))
        cost = 0
        for j in range(len(x_arr)):
            a = np.dot(x_arr[j], w)
            a = sigmoid(a)
            grad += a * (1 - a) * (a - y_arr[j]) * x_arr[j]
            # calculating cost
            cost += ((y_arr[j] - a)**2)
        w = w - (learning_rate * grad) / m
        print("iteration: {}, cost: {}".format(i, cost))
    costs.append(cost)
    return w


# running the gradient descent algorithm for double-leayer network
def gradient_descent_nonlinear(train_data, learning_rate=0.01, epoch=5000):
    m = len(train_data)
    x_arr = [np.append(np.array(1), point.values) for point in train_data]
    y_arr = [np.array(point.label) for point in train_data]
    '''w = np.random.normal(loc=0.0, scale=0.1, size=len(x_arr[0]))
    u = np.random.normal(loc=0.0, scale=0.1, size=len(x_arr[0]))
    v = np.random.normal(loc=0.0, scale=0.1, size=len(x_arr[0]))'''
    w = np.append(np.array([0]), np.random.normal(loc=0.0, scale=0.1, size=len(x_arr[0]) - 1))
    u = np.append(np.array([0]), np.random.normal(loc=0.0, scale=0.1, size=len(x_arr[0]) - 1))
    v = np.append(np.array([0]), np.random.normal(loc=0.0, scale=0.1, size=len(x_arr[0]) - 1))
    costs = []
    for i in range(epoch):
        grad_v = np.zeros(len(w))
        grad_u = np.zeros(len(w))
        grad_w = np.zeros(len(w))
        cost = 0
        for j in range(len(x_arr)):
            # calculating second layer gradients
            a_arr = np.vstack((np.array(1), np.dot(x_arr[j], w), np.dot(x_arr[j], u)))
            a_arr = a_arr.T[0]
            a2 = np.dot(a_arr, v)
            a2 = sigmoid(a2)
            grad_v += (a2 - y_arr[j]) * a2 * (1 - a2) * a_arr

            # calculating first layaer gradients
            a1 = np.dot(x_arr[j], u)
            a1 = sigmoid(a1)
            grad_u += (a2 - y_arr[j]) * a2 * (1 - a2) * a1 * (1 - a1) * v[2] * x_arr[j]

            a0 = np.dot(x_arr[j], w)
            a0 = sigmoid(a0)
            grad_w += (a2 - y_arr[j]) * a2 * (1 - a2) * a0 * (1 - a0) * v[1] * x_arr[j]

            # calculating cost
            cost += ((y_arr[j] - a2) ** 2)

        v = v - (learning_rate * grad_v) / m
        u = u - (learning_rate * grad_u) / m
        w = w - (learning_rate * grad_w) / m
        costs.append(cost)
        print("iteration: {}, cost: {}".format(i, cost))
    return v, u, w


# predict the result for test data - linear
def predict_result_linear(test_data, w):
    data = copy.deepcopy(test_data)
    x_arr = [np.append(np.array(1), point.values) for point in test_data]
    for i in range(len(x_arr)):
        res = np.dot(x_arr[i], w)
        if res >= 0.5:
            data[i].predicted_label = 1
        else:
            data[i].predicted_label = 0
    return data


# predict the result for test data - Non-linear
def predict_result_nonlinear(test_data, v, u, w):
    data = copy.deepcopy(test_data)
    x_arr = [np.append(np.array(1), point.values) for point in test_data]
    for i in range(len(x_arr)):
        a_arr = np.vstack((np.array(1), np.dot(x_arr[i], w), np.dot(x_arr[i], u)))
        a_arr = a_arr.T[0]
        res = np.dot(a_arr, v)
        if res >= 0.5:
            data[i].predicted_label = 1
        else:
            data[i].predicted_label = 0
    return data


# calculating the accuracy of the predicted model
def calculate_accuracy(predicted_model):
    accuracy = max_accuracy = float(len(predicted_model))
    for point in predicted_model:
        if point.predicted_label != point.label:
            accuracy -= 1
    return accuracy / max_accuracy




# reading the dataset from file
def init_file(file_name):
    points = []
    with open(file_name, 'r') as reader:
        next(reader)
        for line in reader:
            values = [float(x) for x in line.split(',')]
            label = values[-1]
            values = values[:-1]
            points.append(Point(values, label))
    return points


# plotting the given array of points based on their label, showing each class with a different color
def plot_data(points, title):
        colors = ["blue", "red", "green", "yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"]
        classes = [[], []]
        # separating data with different labels into different classes
        for i in range(len(points)):
            if points[i].label == 0:
                classes[0].append(points[i])
            else:
                classes[1].append(points[i])
        # plotting each class with different color
        for i in range(len(classes)):
            for j in range(len(classes[i])):
                x_values = [y.values[0] for y in [x for x in classes[i]]]
                y_values = [y.values[1] for y in [x for x in classes[i]]]
                plt.scatter(x_values, y_values, c=colors[i])
        plt.title(title)
        plt.show()


# plotting the given data based on their predicted_label
def plot_test_data(points, title):
        colors = ["blue", "red","black", "green", "yellow","pink","orange","purple","beige","brown","gray","cyan","magenta"]
        classes = [[], [], []] # [[predicted_label = 0.0] [predicted_label =1.0] [predicted_label = None]]
        # separating data with different labels into different classes
        for i in range(len(points)):
            if points[i].predicted_label == 0:
                classes[0].append(points[i])
            elif points[i].predicted_label == 1:
                classes[1].append(points[i])
            else:
                classes[2].append(points[i])
        # plotting each class with different color
        for i in range(len(classes)):
            for j in range(len(classes[i])):
                x_values = [y.values[0] for y in [x for x in classes[i]]]
                y_values = [y.values[1] for y in [x for x in classes[i]]]
                plt.scatter(x_values, y_values, c=colors[i])
        plt.title(title)
        plt.show()


# divide the data into training data and test data
def divide_data(points):
    points_copy = copy.deepcopy(points)
    np.random.shuffle(points_copy)
    train_len = int(0.75*len(points_copy))
    train_data = points_copy[:train_len]
    test_data = points_copy[train_len:]
    return train_data, test_data
