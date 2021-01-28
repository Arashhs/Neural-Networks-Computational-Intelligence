import neural_network as neural

learning_rate = 0.01
epoch = 1000
m = None

def main():
    global m
    # initializing dataset
    points = neural.init_file("dataset.csv")
    # neural.plot_data(points, "All data")
    train_data, test_data = neural.divide_data(points)
    # neural.plot_data(train_data, "Train data")
    # neural.plot_test_data(test_data, "Test data")
    m = len(train_data)

    print("done")

if __name__ == '__main__':
    main()