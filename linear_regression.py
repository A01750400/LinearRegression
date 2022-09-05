def gradient_descent(curr_m, curr_b, data, alfa):
    m_gradient = 0
    b_gradient = 0

    n = len(data)

    for i in range(n):
        x = data.iloc[i].X
        y = data.iloc[i].Y

        m_gradient += -(2/n) * x * (y - (curr_m * x + curr_b))
        b_gradient += -(2/n) * (y - (curr_m * x + curr_b))

    m = curr_m - m_gradient * alfa
    b = curr_b - b_gradient * alfa
    return m, b
