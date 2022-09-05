import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import gradient_descent

if __name__ == "__main__":
    data = pd.read_csv("slr02.csv")
    print(data)

    m = 0
    b = 0
    alfa = 0.001
    epochs = 10

    for i in range(epochs):
        m,b = gradient_descent(m, b, data, alfa)

    print(m, b)
    plt.scatter(data.X, data.Y)
    plt.xlabel("chirps/sec for the stripped ground cricket")
    plt.ylabel("temperature (F)")
    plt.plot(list(range(10, 25)), [m * x + b for x in range(10,25)])
    plt.show()
