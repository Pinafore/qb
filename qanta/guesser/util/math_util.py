from numpy import *

# derivative of tanh
def dtanh0(x):
	return 1 - square(x)


# derivative of normalized tanh
def dtanh(x):
    norm = linalg.norm(x)
    y = x - power(x, 3)
    dia = diag((1 - square(x)).flatten()) / norm
    pro = y.dot(x.T) / power(norm, 3)
    out = dia - pro
    return out


## other utility functions not used here (but experimented with!)
def softmax(w):
    ew = exp(w - max(w))
    return ew / sum(ew)

def sigmoid(w):
    sm = 1 / (1 + exp(-w))
    return sm

def d_sigmoid(w):
    return w * (1 - w)

def relu(x):
    return x * (x > 0)

def drelu(x):
    return x > 0
    
def crossent(label, classification):
    return -sum(label * log(classification))

def dcrossent(label, classification):
    return classification - label

def square_loss(label, classification):
    err = label - classification
    return 0.5 * err.T.dot(err)