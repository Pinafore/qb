def relu(x):
    return x * (x > 0)


def drelu(x):
    return x > 0