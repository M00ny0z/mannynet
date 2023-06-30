from collections import defaultdict
from mynlplib.clf_base import predict, make_feature_vector

# deliverable 4.1
def perceptron_update(single_x, single_y, weights, labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''
    f_x_y = make_feature_vector(single_x, single_y)
    y_hat = predict(single_x, weights, labels)[0]
    f_y_hat = make_feature_vector(single_x, y_hat)
    if (y_hat != single_y):
        return feature_vec_diff(f_x_y, f_y_hat)
    else:
        return defaultdict(float)

# deliverable 4.2
def estimate_perceptron(x_vector, y_vector, N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''
    my_labels = set(y_vector)
    my_weights = defaultdict(float)
    weight_history = []
    for it in range(N_its):
        for x_i,y_i in zip(x_vector, y_vector):
            # YOUR CODE GOES HERE
            weight_updates = perceptron_update(x_i, y_i, my_weights, my_labels)
            if (len(weight_updates) != 0):
                add_weight_vectors(my_weights, weight_updates)
        weight_history.append(my_weights.copy())
    return my_weights, weight_history

def add_weight_vectors(original_weights, weight_updates):
    super_set = set(original_weights.keys()).union(set(weight_updates.keys()))
    for key in super_set:
        original_weights[key] = original_weights[key] + weight_updates[key]

def feature_vec_diff(vector_a, vector_b):
    '''
    computes vector_a - vector_b
    '''
    output_vector = defaultdict(float)
    super_set = set(vector_a.keys()).union(set(vector_b.keys()))
    for key in super_set:
        if (vector_a[key] - vector_b[key] != 0):
            output_vector[key] = vector_a[key] - vector_b[key]
    return output_vector