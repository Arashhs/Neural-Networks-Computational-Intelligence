import neural_network as neural

learning_rate = 0.015
epoch = 3000
m = None

def main():
    global m
    # initializing dataset
    points = neural.init_file("dataset.csv")
    # neural.plot_data(points, "All data")
    train_data, test_data = neural.divide_data(points)
    neural.plot_data(train_data, "Train data")
    neural.plot_test_data(test_data, "Test data")
    m = len(train_data)

    # running gradient descent
    w = neural.gradient_descent_linear(train_data, learning_rate, epoch)
    print(w)
    result = neural.predict_result_linear(test_data, w)
    neural.plot_test_data(result, "Prediction For Test Data")
    print(neural.calculate_accuracy(result))

    print("done")

if __name__ == '__main__':
    main()