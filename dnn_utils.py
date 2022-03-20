import math
import numpy as np


def sigmoid(Z: np.ndarray):
    """Implements the sigmoid activation function.

    This implements the logistic function. Given the independent variable Z,
    the return value A is calculated by the formula `A = 1/( 1+e^(-Z) )`.

    Arguments:
    Z -- Output of the linear layer, numpy array of any shape.
    
    Returns:
    A -- Output of sigmoid(Z), same shape as Z.
    cache -- Cache of Z, stored for computing the backward propagation efficiently.
    """

    A = 1/(1 + np.exp(-Z))
    cache = Z

    assert(A.shape == Z.shape)

    return A, cache

def relu(Z: np.ndarray):
    """Implements the RELU activation function.

    This implements the ReLU (Rectified Linear Unit) activation function. Given the independent
    variable Z, the return value A is calculated by the formula `A = max(0, Z)`.

    Arguments:
    Z -- Output of the linear layer, numpy array of any shape.

    Returns:
    A -- Output of relu(z), of the same shape as Z.
    cache -- Cache of Z, stored for computing the backward propagation efficiently.
    """

    A = np.maximum(0, Z)
    cache = Z

    assert(A.shape == Z.shape)

    return A, cache

def sigmoid_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """Implement the backward propagation for a single SIGMOID unit.

    `s = A = 1/(1 + e^(-Z) )`. Given value of dA, this function computes dZ by the formula
    `dZ = dA * s * (1-s)`.

    Arguments:
    dA -- Gradient of the cost with respect to post-activation A.
    cache -- 'Z' where we store for computing backward propagation efficiently.

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert(dZ.shape == Z.shape)

    return dZ

def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- Gradient of the cost with respect to post-activation A.
    cache -- 'Z' where we store for computing backward propagation efficiently.

    Returns:
    dZ -- Gradient of the cost with respect to Z.
    """

    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert(dZ.shape == Z.shape)

    return dZ

def initialize_parameters_deep(layers_dims: tuple) -> dict:
    """Initialize parameters for every layer.

    Arguments:
    layers_dims -- Python tuple (or list) containing the dimensions of each layer in our network.
    
    Returns:
    parameters -- Python dictionary containing your parameters ("W1", "b1", ..., "WL", "bL":
                  Wl - weight matrix of shape (layers_dims[l], layers_dims[l-1])
                  bl - bias vector of shape (layers_dims[l], 1) ).
    """

    np.random.seed(1) 
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))

    return parameters

def linear_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray):
    """Implement the linear part of a layer's forward propagation.

    Calculation formula is `Z = W dot A_prev + b`.

    Arguments:
    A_prev -- Activations from previous layer (or input data): shape = (size of previous layer, number of examples).
    W -- Weights matrix: shape = (size of current layer, size of previous layer).
    b -- Bias vector: shape = (size of current layer, 1).

    Returns:
    Z -- The input of the activation function, also called pre-activation parameter.
    cache -- A python tuple containing 'A', 'W', and 'b', stored for computing the backward pass efficiently.
    """

    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    return Z, cache

def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, 
                              activation: str, keep_prob: float = 1.0):
    """Implement the forward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    A_prev -- Activations from previous layer (or input data): (size of previous layer, number of examples).
    W -- Weights matrix, numpy array of shape (size of current layer, size of previous layer).
    b -- Bias vector, numpy array of shape (size of the current layer, 1).
    activation -- The activation to be used in this layer, stored as a text string: "sigmoid" or "relu".
    keep_prob -- Probability of keeping a neuron active during drop-out, scalar.

    Returns:
    A -- The output of the activation function, also called the post-activation value.
    cache -- A python dictionary containing "linear_cache", "activation_cache" and "dropout_cache;
             stored for computing the backward pass efficiently.
    """

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    D = np.random.rand(*A.shape) < keep_prob
    A = A * D / keep_prob

    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache, D)

    return A, cache 

def linear_activation_forward_check(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, 
                                    activation: str, drop_cache: np.ndarray, keep_prob: float = 1.0):
    """Implement the forward propagation for the LINEAR->ACTIVATION layer, but for checking.

    Arguments:
    A_prev -- Activations from previous layer (or input data): shape = (size of previous layer, number of examples).
    W -- Weights matrix, numpy array of shape (size of current layer, size of previous layer).
    b -- Bias vector, numpy array of shape (size of the current layer, 1).
    activation -- The activation to be used in this layer, stored as a text string: "sigmoid" or "relu".
    drop_cache -- Dropout matrix, numpy array of shape (size of current layer, number of examples).
    keep_prob -- Probability of keeping a neuron active during drop-out, scalar.

    Returns:
    A -- The output of the activation function, also called the post-activation value.
    """

    if activation == 'sigmoid':
        Z, _ = linear_forward(A_prev, W, b)
        A, _ = sigmoid(Z)
    
    elif activation == 'relu':
        Z, _ = linear_forward(A_prev, W, b)
        A, _ = relu(Z)
    
    A = A * drop_cache / keep_prob

    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    return A

def L_model_forward(X: np.ndarray, parameters: dict, keep_prob: float = 1.0):
    """Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
    
    Arguments:
    X -- Input Data, numpy array of shape (input size, number of examples).
    parameters -- Python dictonary containing weigh and bias matrix "W1", "b1", ..., "WL", "bL".
    keep_prob -- Probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    AL -- Last post-activation value.
    caches -- List of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1).
    """

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, 
                                                parameters['W' + str(l)], 
                                                parameters['b' + str(l)],
                                                activation='relu',
                                                keep_prob=keep_prob)
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)],
                                            parameters['b' + str(L)],
                                            activation='sigmoid',
                                            keep_prob=1.0)
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

def L_model_forward_check(X: np.ndarray, parameters: dict, drop_caches: list, keep_prob: float = 1.0):
    """Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation, but for checking.
    
    Arguments:
    X -- Input Data, numpy array of shape (input size, number of examples).
    parameters -- Python dictonary containing weigh and bias matrix "W1", "b1", ..., "WL", "bL".
    drop_caches -- Dropout caches, containing dropout mask of each layer.
    keep_prob -- Probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    AL -- Last post-activation value.
    """

    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A = linear_activation_forward_check(A_prev, 
                                                parameters['W' + str(l)], 
                                                parameters['b' + str(l)],
                                                activation='relu',
                                                drop_cache = drop_caches[l-1],
                                                keep_prob=keep_prob)
    
    AL = linear_activation_forward_check(A, parameters['W' + str(L)],
                                            parameters['b' + str(L)],
                                            activation='sigmoid',
                                            drop_cache = drop_caches[L-1],
                                            keep_prob=1.0)

    assert(AL.shape == (1, X.shape[1]))

    return AL

def compute_cost(AL: np.ndarray, Y:np.ndarray, parameters: dict, lambd: float = 0) -> np.ndarray:
    """Implement the cost function.

    Arguments:
    AL -- Probability vector corresponding to your label predictions, shape of (1, number of examples).
    Y -- True "label" vector, shape of (1, number of examples).
    parameters -- Python dictonary containing weigh and bias matrix "W1", "b1", ..., "WL", "bL".
    lambd -- L2 Regularization hyperparameter, scalar.

    Returns:
    cost -- Cross-entropy cost.
    """

    m = Y.shape[1]
    L = len(parameters) // 2 + 1
    Wl = [parameters['W' + str(l)] for l in range(1, L)]

    cross_entropy_cost = -1./m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL), axis=1, keepdims=True) 
    L2_regularization_cost =  sum(np.sum(np.square(w)) for w in Wl) * lambd / 2 / m
    cost = cross_entropy_cost + L2_regularization_cost
    cost = np.squeeze(cost)
    
    assert(cost.shape == ())

    return cost

def linear_backward(dZ: np.ndarray, cache, lambd: float = 0):
    """Implement the linear portion of backward propagation for a single layer.

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l).
    cache -- Tuple of values (A_prev, W, b) coming from the forward propagation in the current layer.
    lambd -- L2 Regularization hyperparameter, scalar.

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev.
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W.
    db -- Gradient of the cost with respect to b (current layer l), same shape as b.
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ, A_prev.T) + W * lambd / m
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA: np.ndarray, cache, activation: str, lambd: float=0, keep_prob: float = 1.0):
    """Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- Post-activation gradient for current layer l.
    cache -- Tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently.
    activation -- The activation to be used in this layer, stored as a text string: "sigmoid" or "relu".
    lambd -- L2 Regularization hyperparameter, scalar.
    keep_prob -- Probability of keeping a neuron active during drop-out, scalar.

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev.
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W.
    db -- Gradient of the cost with respect to b (current layer l), same shape as b.
    """

    linear_cache, activation_cache, D = cache

    dA = dA * D / keep_prob

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd=lambd)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd=lambd)

    return dA_prev, dW, db

def L_model_backward(AL: np.ndarray, Y: np.ndarray, caches, lambd: float = 0, keep_prob: float = 1.0):
    """Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group.
    
    Arguments:
    AL -- Probability vector, output of the forward propagation (L_model_forward()).
    Y -- True "label" vector.
    caches -- List of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2).
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1).
    lambd -- L2 Regularization hyperparameter, scalar.
    keep_prob -- Probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """

    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL)) 

    current_cache = caches[L-1]
    dA, \
    grads['dW' + str(L)], \
    grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, activation='sigmoid', lambd=lambd, keep_prob=1.0)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA, \
        grads['dW' + str(l+1)], \
        grads['db' + str(l+1)]  = linear_activation_backward(dA, current_cache, 
                                                             activation='relu', lambd=lambd, keep_prob=keep_prob)

    return grads

def update_parameters(parameters: dict, grads: dict, learning_rate: float):
    """Update parameters using gradient descent.
    
    Arguments:
    parameters -- Python dictionary containing parameters.
    grads -- Python dictionary containing gradients, output of L_model_backward.
    learning_rate -- learning rate of the gradient descent update rule.
    
    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - grads['dW' + str(l+1)] * learning_rate
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - grads['db' + str(l+1)] * learning_rate
    
    return parameters

def predict(X: np.ndarray, Y: np.ndarray, parameters: dict):
    """This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- Input Data set of examples you would like to label.
    Y -- True "label" vector.
    parameters -- Parameters of the trained model.
    
    Returns:
    p -- Predictions for the given dataset X.
    accuracy -- Accuracy of prediction by this neural network.
    """

    m = X.shape[1]
    p = np.zeros((1, m))

    probas, _ = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1 
        else:
            p[0, i] = 0
    
    accuracy = np.sum((p == Y)/m)

    return p, accuracy

def identify(X: np.ndarray, parameters: dict):
    """Identification by input data.

    Arguments:
    X -- Input Data set of examples you would like to label.
    parameters -- Parameters of the trained model.

    Returns:
    p -- Predictions (or identification) for the given dataset X.
    """
    m = X.shape[1]
    p = np.zeros((1, m))

    probs, _ = L_model_forward(X, parameters)

    p[0, :] = 1 if probs[0, :] > 0.5 else 0

    return p

def L_layer_model(X: np.ndarray, Y: np.ndarray, layers_dims: list, learning_rate: float = 0.0075, num_iterations: int = 3000, 
                  lambd: float = 0 , keep_prob: float = 1.0, print_cost: bool = True, check_back_prop: bool = False, 
                  step: int = 100, use_mini_batch: bool = False, mini_batch_size: int = 64):
    """Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- Data, of shape (input size, number of examples).
    Y -- True "label" vector, of shape (1, number of examples).
    layers_dims -- List containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- Learning rate of the gradient descent update rule.
    num_iterations -- Number of iterations of the optimization loop.
    lambd -- L2 regularization hyperparameter, scalar.
    keep_prob -- Probability of keeping a neuron active during drop-out, scalar
    print_cost -- If True, it prints the cost every xxx steps.
    check_back_prop -- Whether to check the correctness of back-propagation.
    step -- The step size of recording cost and printing cost.
    use_mini_batch -- Whether to use mini-batch gradient descent.
    mini_batch_size -- The size of a mini batch.
    
    Returns:
    parameters -- Parameters learnt by the model. They can then be used to predict.
    costs -- Costs learnt by the model. They can then be used to plot a cost curve.
    """

    np.random.seed(1)
    costs = []                    
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # deal with mini_batch gradient descent
    if use_mini_batch:
        mini_batches = random_mini_batches(X, Y, mini_batch_size)
        k = 1
        for mini_batch in mini_batches:
            X_mini, Y_mini = mini_batch
            print('mini batch gradient descent {}...'.format(k))
            k += 1

            # Loop (gradient descent)
            for i in range(0, num_iterations):
                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                AL, caches = L_model_forward(X_mini, parameters, keep_prob=keep_prob)
                
                # Compute cost.
                cost = compute_cost(AL, Y_mini, parameters ,lambd=lambd)
            
                # Backward propagation.
                grads = L_model_backward(AL, Y_mini, caches, lambd=lambd, keep_prob=keep_prob)
        
                # Gradient check
                if check_back_prop:
                    drop_caches = [caches[i][2] for i in range(len(caches))]
                    gradient_check(parameters, grads, drop_caches, X_mini, Y_mini, layers_dims, lambd, keep_prob)

                # Update parameters.
                parameters = update_parameters(parameters, grads, learning_rate)

                # Print and Save the cost every 100 training example
                if print_cost and i == 0:
                    print("Cost after iteration {:d}: {:f}".format(i+1, cost))
                    costs.append(cost)
                if print_cost and (i+1) % step == 0:
                    print("Cost after iteration {:d}: {:f}".format(i+1, cost))
                    costs.append(cost)
    else:
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(X, parameters, keep_prob=keep_prob)
            
            # Compute cost.
            cost = compute_cost(AL, Y, parameters ,lambd=lambd)
        
            # Backward propagation.
            grads = L_model_backward(AL, Y, caches, lambd=lambd, keep_prob=keep_prob)
    
            # Gradient check
            if check_back_prop:
                drop_caches = [caches[i][2] for i in range(len(caches))]
                gradient_check(parameters, grads, drop_caches, X, Y, layers_dims, lambd, keep_prob)

            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)

            # Print and Save the cost every 100 training example
            if print_cost and i == 0:
                print("Cost after iteration {:d}: {:f}".format(i+1, cost))
                costs.append(cost)
            if print_cost and (i+1) % step == 0:
                print("Cost after iteration {:d}: {:f}".format(i+1, cost))
                costs.append(cost)
    
    return parameters, costs

def dictionary_to_vector(parameters: dict):
    """Roll all our parameters dictionary into a single vector satisfying our specific required shape.

    Arguments:
    parameters -- Python dictonary containing weigh and bias matrix "W1", "b1", ..., "WL", "bL".

    Returns:
    theta -- Vector containing all parameters.
    """

    L = len(parameters) // 2
    theta = None
    for l in range(L):
        new_vector_w = parameters[f"W{l+1}"].reshape((-1, 1))
        new_vector_b = parameters[f"b{l+1}"].reshape((-1, 1))
        new_vector = np.concatenate((new_vector_w, new_vector_b), axis=0)
        if theta is None:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
    
    return theta

def vector_to_dictionary(theta: np.ndarray, layers_dims: dict):
    """Unroll all our parameters dictionary from a single vector satisfying our specific required shape.

    Arguments:
    theta -- Vector containing all parameters.
    layers_dims -- List containing the input size and each layer size, of length (number of layers + 1).

    Returns:
    parameters -- Python dictonary containing weigh and bias matrix ("W1", "b1", ..., "WL", "bL"). 
    """

    parameters = {}
    L = len(layers_dims)
    
    start = 0
    for l in range(1, L):
        stop = start + layers_dims[l] * layers_dims[l-1]
        parameters[f"W{l}"] = theta[start:stop].reshape((layers_dims[l], layers_dims[l-1]))
        start = stop
        stop = start + layers_dims[l]
        parameters[f"b{l}"] = theta[start:stop].reshape((layers_dims[l], 1))
        start = stop
    
    return parameters

def grads_to_vector(grads: dict):
    """Roll all our grads dictionary into a single vector satisfying our specific required shape.

    Arguments:
    grads -- Python dictonary containing gradients of cost with respect to the parameters("dW1", "db1", ..., "dWL", "dbL").

    Returns:
    theta -- Vector containing all grads.
    """

    L = len(grads) // 2
    theta = None
    for l in range(L):
        new_vector_dw = grads[f"dW{l+1}"].reshape((-1, 1))
        new_vector_db = grads[f"db{l+1}"].reshape((-1, 1))
        new_vector = np.concatenate((new_vector_dw, new_vector_db), axis=0)
        if theta is None:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
    
    return theta

def gradient_check(parameters: dict, grads: dict, drop_caches: list, X: np.ndarray, Y: np.ndarray, layers_dims, 
                   lambd: float = 0, keep_prob = 1.0, epsilon: float = 1e-7):
    """Checks if backward_propagation_n computes correctly the gradient of the cost output by L_model_forward.
    
    Arguments:
    parameters -- Python dictonary containing weigh and bias matrix "W1", "b1", ..., "WL", "bL".
    grads -- Output of L_model_backward, contains gradients of the cost with respect to the parameters. 
    drop_caches -- Dropout caches, containing dropout mask of each layer.
    X -- Data, numpy array of shape (input size, number of examples).
    Y -- True "label".
    layers_dims -- List containing the input size and each layer size, of length (number of layers + 1).
    lambd -- L2 regularization hyperparameter, scalar.
    keep_prob -- Probability of keeping a neuron active during drop-out, scalar.
    epsilon -- Tiny shift to the input to compute approximated gradient.
    
    Returns:
    difference -- Difference between the approximated gradient and the backward propagation gradient.
    """

    parameters_values = dictionary_to_vector(parameters)
    grad = grads_to_vector(grads)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    grad_approx = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        theta_plus = np.copy(parameters_values)
        theta_plus[i][0] += epsilon
        theta_plus = vector_to_dictionary(theta_plus, layers_dims)
        AL = L_model_forward_check(X, theta_plus, drop_caches, keep_prob)
        J_plus[i] = compute_cost(AL, Y, theta_plus, lambd=lambd)

        theta_minus = np.copy(parameters_values)
        theta_minus[i][0] -= epsilon
        theta_minus = vector_to_dictionary(theta_minus, layers_dims)
        AL = L_model_forward_check(X, theta_minus, drop_caches, keep_prob)
        J_minus[i] = compute_cost(AL, Y, theta_minus, lambd=lambd)

        grad_approx[i] = (J_plus[i] - J_minus[i]) / 2 / epsilon

    numerator = np.linalg.norm(grad - grad_approx)
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)
    difference = numerator / denominator

    if difference > 1e-7:
        print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """Creates a list of random minibatches from (X, Y).

    Arguments:
    X -- Input data, of shape (input size, number of examples).
    Y -- True "lable" vector, of shape (1, number of examples).
    mini_batch_size -- Size of the mini batches size.
    seed -- Random seed.

    Returns:
    mini_batches -- List of synchronous (mini_batch_X, mini_batch_Y).
    """

    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # shuffle (X, Y)
    permutaion = list(np.random.permutation(m))
    shuffle_X = X[:, permutaion]
    shuffle_Y = Y[:, permutaion]

    # partition (shuffle_X, shuffle_Y), except the end case
    num_of_mini_batches = math.floor(m/mini_batch_size) 
    for i in range(num_of_mini_batches):
        mini_batch_X = shuffle_X[:, mini_batch_size*i:mini_batch_size*(i+1)]
        mini_batch_Y = shuffle_Y[:, mini_batch_size*i:mini_batch_size*(i+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # deal with end case
    if m % mini_batch_size != 0:
        mini_batch_X = shuffle_X[:, mini_batch_size*num_of_mini_batches:m]
        mini_batch_Y = shuffle_Y[:, mini_batch_size*num_of_mini_batches:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches