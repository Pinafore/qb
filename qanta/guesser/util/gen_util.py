from numpy import *


def unroll_params(arr, d, len_voc, deep=1):
    mat_size = d * d
    ind = 0

    params = []
    for i in range(0, deep):
        params.append(arr[ind: ind + mat_size].reshape((d, d)))
        ind += mat_size
        params.append(arr[ind: ind + d].reshape((d, 1)))
        ind += d

    params.append(arr[ind: ind + len_voc * d].reshape((d, len_voc)))
    return params


# roll all parameters into a single vector
def roll_params(params):
    return concatenate([p.ravel() for p in params])


# initialize all parameters to magic
def init_params(d, deep=1):
    params = []
    magic_number = 0.08
    for i in range(0, deep):
        params.append((random.rand(d, d) * 2 - 1) * magic_number)
        params.append((random.rand(d, 1) * 2 - 1) * magic_number)

    return params


# returns list of zero gradients which backprop modifies
def init_grads(d, len_voc, deep=1):

    grads = []
    for i in range(0, deep):
        grads.append(zeros((d, d)))
        grads.append(zeros((d, 1)))

    grads.append(zeros((d, len_voc)))
    return grads


# random embedding matrix for gradient checks
def gen_rand_we(len_voc, d):
    r = sqrt(6) / sqrt(257)
    we = random.rand(d, len_voc) * 2 * r - r
    return we