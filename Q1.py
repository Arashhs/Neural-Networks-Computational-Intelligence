import neural_network as neural


def main():
    # initializing dataset
    points = neural.init_file("dataset.csv")
    neural.plot_data(points, "All data")
    train_data, test_data = neural.divide_data(points)
    # neural.plot_data(train_data, "Train data")
    # neural.plot_test_data(test_data, "Test data")

    print("done")

if __name__ == '__main__':
    main()