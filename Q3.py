import neural_network as neural

learning_rate = 0.08
epoch = 10000

def main():
    # initializing dataset
    print("--- Double-Layer Neural Network ---")
    points = neural.init_file("dataset.csv")
    # neural.plot_data(points, "All data")
    train_data, test_data = neural.divide_data(points)
    neural.plot_data(train_data, "Train data")
    neural.plot_test_data(test_data, "Test data")

    # running gradient descent algorithm
    print("Running the gradient descent algorithm; Please be patient!")
    v, u, w = neural.gradient_descent_nonlinear(train_data, learning_rate, epoch)
    print("Edge Weights:\nV: {}\nU: {}\nW: {}".format(v, u, w))

    # predicting the label for test data
    result = neural.predict_result_nonlinear(test_data, v, u, w)
    neural.plot_test_data(result, "Prediction For Test Data")

    # calculating the accuracy of the predicted model
    accuracy = neural.calculate_accuracy(result)
    print("\nAccuracy:", accuracy)

    print("\n All Done!")

if __name__ == '__main__':
    main()