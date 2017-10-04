import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    N = data.shape[0]
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Dy * N
    y = labels.transpose()
    # Dx * N
    a_0 = data.transpose()
    # H * N
    b_0 = np.dot(b1.transpose(), np.ones((1, N)))
    # Dy * N
    b_1 = np.dot(b2.transpose(), np.ones((1, N)))
    # H * Dx
    w_0 = W1.transpose()
    # Dy * H
    w_1 = W2.transpose()

    ### YOUR CODE HERE: forward propagation
    # H * N
    z_0 = np.dot(w_0, a_0) + b_0
    print("z_0: [{}, {}]".format(np.min(z_0), np.max(z_0)))
    a_1 = sigmoid(z_0)
    print("a_1: [{}, {}]".format(np.min(a_1), np.max(a_1)))

    # Dy * N
    z_1 = np.dot(w_1, a_1) + b_1
    print("z_1: [{}, {}]".format(np.min(z_1), np.max(z_1)))
    a_2 = sigmoid(z_1)
    cost = np.sum(np.multiply(y, np.log(a_2))) / N
    print("cost: {}".format(cost))
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    # Dy, N
    g_z_1 = np.multiply(y, 1 - a_2)
    assert g_z_1.shape == (Dy, N)
    print("g_z_1: [{}, {}]".format(np.min(g_z_1), np.max(g_z_1)))

    # Dy * H
    g_w_1 = np.dot(g_z_1, a_1.transpose())
    print("g_w_1: [{}, {}]".format(np.min(g_w_1), np.max(g_w_1)))
    assert g_w_1.shape == (Dy, H)
    gradW2 = g_w_1.transpose() / N

    gradb2 = np.sum(g_z_1, 1) / N

    # H * N 
    g_z_0 = np.multiply(np.multiply(a_1, 1 - a_1), 
        np.dot(w_1.transpose(), g_z_1))
    assert g_z_0.shape == (H, N)
    print("g_z_0: [{}, {}]".format(np.min(g_z_0), np.max(g_z_0)))

    # H * Dx
    g_w_0 = np.dot(g_z_0, a_0.transpose())
    assert g_w_0.shape == (H, Dx)
    print("g_w_0: [{}, {}]".format(np.min(g_w_0), np.max(g_w_0)))

    gradW1 = g_w_0.transpose() / N

    gradb1 = np.sum(g_z_0, 1) / N

    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
