import numpy as np
from rnn_utils import *


"""
Instructions:

    1) Compute the hidden state with tanh activation: a⟨t⟩=tanh(Waaa⟨t−1⟩+Waxx⟨t⟩+ba)a⟨t⟩=tanh⁡(Waaa⟨t−1⟩+Waxx⟨t⟩+ba).
    2) Using your new hidden state a⟨t⟩a⟨t⟩, compute the prediction ŷ ⟨t⟩=softmax(Wyaa⟨t⟩+by)y^⟨t⟩=softmax(Wyaa⟨t⟩+by). We provided you a function: softmax.
    3) Store (a⟨t⟩,a⟨t−1⟩,x⟨t⟩,parameters)(a⟨t⟩,a⟨t−1⟩,x⟨t⟩,parameters) in cache
    4) Return a⟨t⟩a⟨t⟩ , y⟨t⟩y⟨t⟩ and cache

We will vectorize over mm examples. Thus, x⟨t⟩ will have dimension (nx,m)(nx,m), and a⟨t⟩ will have dimension (na,m). 
"""

def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # compute next activation state using the formula given above
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    # print()
    # print("a_prev: ", a_prev.shape)
    # print("xt: ", xt.shape)
    # print("Wax: ", Wax.shape)
    # print("Waa: ", Waa.shape)
    # print("Wya: ", Wya.shape)
    # print("ba: ", ba.shape)
    # print("by: ", by.shape)
    # print()
    # print("a_next: ", a_next.shape)
    # print("yt_pred: ", yt_pred.shape)
    return a_next, yt_pred, cache


# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

# # rnn_cell_forward(xt, a_prev, parameters)

# a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", a_next.shape)
# print("yt_pred[1] =", yt_pred[1])
# print("yt_pred.shape = ", yt_pred.shape)

#############################################################


"""
Instructions:
    1) Create a vector of zeros (a) that will store all the hidden states computed by the RNN.
    2) Initialize the "next" hidden state as a0 (initial hidden state).
    3) Start looping over each time step, your incremental index is t :
        - Update the "next" hidden state and the cache by running rnn_cell_forward
        - Store the "next" hidden state in a (t^th position)
        - Store the prediction in y
        - Add the cache to the list of caches
    4) Return a, y and caches
"""

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    
    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # initialize "a" and "y" with zeros (≈2 lines)
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    
    # Initialize a_next (≈1 line)
    a_next = a0
    
    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache (≈1 line)
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[:,:,t] = yt_pred
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)
    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches



# np.random.seed(1)
# x = np.random.randn(3,10,4)
# a0 = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

# rnn_forward(x, a0, parameters)

# a, y_pred, caches = rnn_forward(x, a0, parameters)
# print("a[4][1] = ", a[4][1])
# print("a.shape = ", a.shape)
# print("y_pred[1][3] =", y_pred[1][3])
# print("y_pred.shape = ", y_pred.shape)
# print("caches[1][1][3] =", caches[1][1][3])
# print("len(caches) = ", len(caches))


'''
Instructions:

    1- Concatenate a⟨t−1⟩ and x⟨t⟩ in a single matrix: concat=[a⟨t−1⟩x⟨t⟩]
    2- Compute all the formulas 1-6. You can use sigmoid() (provided) and np.tanh().
    3- Compute the prediction y⟨t⟩. You can use softmax() (provided).
'''

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    ### START CODE HERE ###
    # Concatenate a_prev and xt (≈3 lines)
    concat = None
    concat[: n_a, :] = None
    concat[n_a :, :] = None

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = None
    it = None
    cct = None
    c_next = None
    ot = None
    a_next = None
    
    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = None
    ### END CODE HERE ###

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache
