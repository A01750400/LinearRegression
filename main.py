import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import *

if __name__ == "__main__":
    data = pd.read_csv("slr01.csv")
    #data = pd.read_csv("slr02.csv")
    #data = pd.read_csv("slr03.csv")
    print(data)

    m = 0
    b = 0
    alfa = 0.001
    epochs = 10

    for i in range(epochs):
        m,b = gradient_descent(m, b, data, alfa)

    print(m, b)
    print("Error: ", loss(m, b , data))
    plt.scatter(data.X, data.Y)
    plot_begin = int(data.X.min()) - 2
    plot_end = int(data.X.max()) + 2
    plt.plot(list(range(plot_begin, plot_end)), [m * x + b for x in range(plot_begin, plot_end)])
    plt.show()
