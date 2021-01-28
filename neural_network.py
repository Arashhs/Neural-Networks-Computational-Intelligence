import numpy as np, matplotlib.pyplot as plt
import copy


class Point:
    def __init__(self, values, label):
        self.values = values # values for each point
        self.label = label # label that shows each point belongs to which cluster
        self.predicted_label = None # label that is predicted for test data


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
