#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "dubnet.h"

float logistic(float data) {
    return 1 / (1 + exp(-1 * data));
}


// Run an activation layer on input
// layer l: pointer to layer to run
// tensor x: input to layer
// returns: the result of running the layer y = f(x)
tensor forward_activation_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    ACTIVATION a = l->activation;
    tensor y = tensor_copy(x);

    assert(x.n >= 2);

    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row
    if (a == LOGISTIC || a == RELU || a == LRELU) {
        size_t i;
        size_t len = tensor_len(x);
        for (i = 0; i < len; ++i) {
            switch (a) {
                case LOGISTIC:
                    y.data[i] = logistic(x.data[i]);
                    break;
                case RELU:
                    y.data[i] = x.data[i] > 0 ? x.data[i] : 0;
                    break;
                default:
                    y.data[i] = x.data[i] > 0 ? x.data[i] : 0.01 * x.data[i];
            }
        }
    } else {
        size_t matrix_len = tensor_len(y);
        size_t row_size = y.size[1];
        // Apply e^x op to entire tensor since all values will need it anyways
        tensor_e_x_(y);
        tensor dim_sums = tensor_sum_dim(y, 1);
        // i goes through every row
        size_t i;
        for (i = 0; i < matrix_len; ++i) {
            float row_sum = dim_sums.data[i / row_size];
            y.data[i] = y.data[i] / row_sum;
        }
        tensor_free(dim_sums);
    }

    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
// DONE
tensor backward_activation_layer(layer *l, tensor dy)
{
    tensor x = l->x;
    tensor dx = tensor_copy(dy);
    ACTIVATION a = l->activation;

    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1
    size_t len = tensor_len(x);
    size_t i;
    for (i = 0; i < len; ++i) {
        switch(a) {
            case LOGISTIC:
                dx.data[i] = logistic(x.data[i]) * (1 - logistic(x.data[i])) * dy.data[i];
                break;
            case RELU:
                dx.data[i] = (x.data[i] > 0 ? 1 : 0) * dy.data[i];
                break;
            case LRELU:
                dx.data[i] = (x.data[i] > 0 ? 1 : 0.01) * dy.data[i];
                break;
            default:
                // case is SOFTMAX
                dx.data[i] = dy.data[i];


        }
    }

    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer *l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
