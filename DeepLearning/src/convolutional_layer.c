#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "dubnet.h"
#include "matrix.h"

// Make a column matrix out of an image
// tensor im: image to process
// size_t size: kernel size for convolution operation
// size_t stride: stride for convolution
// size_t pad: # pixels padding on each edge for convolution
// returns: column matrix

tensor im2col(tensor im, size_t size_y, size_t size_x, size_t stride, size_t pad)
{
    assert(im.n == 3);
    size_t i, j, k;

    size_t im_c = im.size[0];
    size_t im_h = im.size[1];
    size_t im_w = im.size[2];

    size_t res_h = (im_h + 2*pad - size_y)/stride + 1;
    size_t res_w = (im_w + 2*pad - size_x)/stride + 1;

    size_t kernel_area = size_y * size_x;
    size_t image_area = im_h * im_w;

    size_t rows = im_c * kernel_area;
    size_t cols = res_w * res_h;

    // insert the values as rows instead of columns
    tensor row_order_output = tensor_vmake(2, cols, rows);

    // Fill in the column matrix with patches from the image
    // i keeps track of which image channel we are looking at
    for (j = 0; j < res_h; ++j) {
        // j is which image level we are looking at
        for (k = 0; k < res_w; ++k) {
            // k is index inside of current image level
            for (i = 0; i < im_c; ++i) {
                // kernel_i keeps track of which row of the kernel we are looking at
                size_t kernel_i;
                for (kernel_i = 0; kernel_i < size_y; ++kernel_i) {
                    // kernel_j keeps track of index inside of row of kernel
                    size_t kernel_j;
                    for (kernel_j = 0; kernel_j < size_x; ++kernel_j) {
                        size_t x_pos = (k * stride) + kernel_j;
                        size_t y_pos = (j * stride) + kernel_i;
                        size_t output_x_pos = (i * kernel_area) + (kernel_i * size_x) + kernel_j;
                        size_t output_y_pos = (j * res_w) + k;
                        size_t output_pos = (output_y_pos * rows) + output_x_pos;
                        float val;
                        // we split like this instead of using int for data pos
                        // to ensure no negative values in a size_t variable
                        // since size_t is unsigned
                        // and ensure we have big enough var to contain all possible sizes
                        if (pad == 0) {
                            size_t data_pos = (y_pos * im_w) + x_pos + (image_area * i);
                            val = im.data[data_pos];
                        } else {
                            int invalidX = x_pos < pad || x_pos >= im_w + pad;
                            int invalidY = y_pos < pad || y_pos >= im_h + pad;
                            if (invalidX || invalidY) {
                                val = 0;
                            } else {
                                x_pos -= pad;
                                y_pos -= pad;
                                size_t data_pos = (y_pos * im_w) + x_pos + (image_area * i);
                                val = im.data[data_pos];
                            }
                        }
                        row_order_output.data[output_pos] = val;
                    }
                }
            }
        }
    }

    tensor col_order_output = matrix_transpose(row_order_output);
    tensor_free(row_order_output);
    return col_order_output;
}

// The reverse of im2col, add elements back into image
// matrix col: column matrix to put back into image
// int size: kernel size
// int stride: convolution stride
// image im: image to add elements back into
tensor col2im(tensor col, size_t c, size_t h, size_t w, size_t size_y, size_t size_x, size_t stride, size_t pad)
{
    tensor row = matrix_transpose(col);
    tensor im = tensor_vmake(3, c, h, w);
    size_t i, j, k, kernel_i;

    size_t im_c = im.size[0];
    size_t im_h = im.size[1];
    size_t im_w = im.size[2];

    size_t res_h = (im_h + 2*pad - size_y)/stride + 1;
    size_t res_w = (im_w + 2*pad - size_x)/stride + 1;

    size_t kernel_area = size_y * size_x;
    size_t image_area = im_w * im_h;

    size_t rows = im_c*size_y*size_x;
    size_t cols = res_w * res_h;
    assert(col.n == 2);
    assert(col.size[0] == rows);
    assert(col.size[1] == cols);

    // Fill in the column matrix with patches from the image
    for (j = 0; j < res_h; ++j) {
        for (k = 0; k < res_w; ++k) {
            for (i = 0; i < im_c; ++i) {
                for (kernel_i = 0; kernel_i < kernel_area; ++kernel_i) {
                    size_t x_pos = kernel_i % size_x;
                    size_t y_pos = kernel_i / size_x;
                    size_t img_x_pos = (k * stride) + x_pos;
                    size_t img_y_pos = (j * stride) + y_pos;
                    // positions for the col
                    size_t input_x_pos = (i * kernel_area) + kernel_i;
                    size_t input_y_pos = (j * res_w) + k;
                    size_t input_pos = (input_y_pos * rows) + input_x_pos;
                    size_t img_pos;
                    if (pad == 0) {
                        img_pos = (img_y_pos * im_w) + img_x_pos + (image_area * i);
                        im.data[img_pos] += row.data[input_pos];
                    } else {
                        int invalidX = img_x_pos < pad || img_x_pos >= im_w + pad;
                        int invalidY = img_y_pos < pad || img_y_pos >= im_h + pad;
                        // if either is invalid, then we dont need to do anything
                        if (!invalidX && !invalidY) {
                            img_x_pos -= pad;
                            img_y_pos -= pad;
                            img_pos = (img_y_pos * im_w) + img_x_pos + (image_area * i);
                            im.data[img_pos] += row.data[input_pos];
                        }
                    }
                }
            }
        }
    }

    tensor_free(row);

    return im;
}

// Run a convolutional layer on input
// layer l: pointer to layer to run
// tensor x: input to layer
// returns: the result of running the layer
tensor forward_convolutional_layer(layer *l, tensor x)
{
    assert(x.n == 4);
    assert(l->w.n == 4);
    assert(x.size[1] == l->w.size[1]); // Same number of channels

    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    size_t im_n = x.size[0];
    // size_t im_c = x.size[1];
    size_t im_h = x.size[2];
    size_t im_w = x.size[3];

    size_t f_n = l->w.size[0];
    size_t f_c = l->w.size[1];
    size_t f_h = l->w.size[2];
    size_t f_w = l->w.size[3];

    size_t y_c = l->w.size[0];
    size_t y_h = (im_h + 2*l->pad - f_h)/l->stride + 1;
    size_t y_w = (im_w + 2*l->pad - f_w)/l->stride + 1;

    tensor y = tensor_vmake(4, im_n, y_c, y_h, y_w);

    // weights in matrix for matrix multiplication
    tensor w = tensor_vview(l->w, 2, f_n, f_c*f_h*f_w);

    size_t i, j;
    for(i = 0; i < x.size[0]; ++i){
        tensor x_i = im2col(tensor_get_(x, i), f_h, f_w, l->stride, l->pad);
        tensor wx = matrix_multiply(w, x_i);
        tensor y_i = tensor_get_(y, i);
        size_t len = tensor_len(wx);
        for(j = 0; j < len; ++j){
            y_i.data[j] = wx.data[j];
        }
        tensor_free(wx);
        tensor_free(x_i);
    }
    tensor b = tensor_vview(l->b, 4, 1, l->b.size[0], 1, 1);
    tensor yb = tensor_add(y, b);

    tensor_free(y);
    tensor_free(b);
    tensor_free(w);

    return yb;
}

// Run a convolutional layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
tensor backward_convolutional_layer(layer *l, tensor dy)
{
    // Calculate dL/db
    tensor db_1 = tensor_sum_dim(dy, 0);
    tensor db_2 = tensor_sum_dim(db_1, 1);
    tensor db = tensor_sum_dim(db_2, 1);
    tensor_axpy_(1, db, l->db);
    tensor_free(db_1);
    tensor_free(db_2);
    tensor_free(db);

    size_t f_n = l->w.size[0];
    size_t f_c = l->w.size[1];
    size_t f_h = l->w.size[2];
    size_t f_w = l->w.size[3];

    tensor x = l->x;

    tensor dx = tensor_make(l->x.n, l->x.size);
    tensor w = tensor_vview(l->w, 2, f_n, f_c*f_h*f_w);
    tensor wt = matrix_transpose(w);

    size_t i;
    for(i = 0; i < x.size[0]; ++i){
        tensor x_i = im2col(tensor_get_(x, i), f_h, f_w, l->stride, l->pad);
        tensor dy_i = tensor_get_(dy, i);
        size_t tmp_size[2];
        tmp_size[0] = dy.size[1];
        tmp_size[1] = dy.size[2]*dy.size[3];
        dy_i.n = 2;
        dy_i.size = tmp_size;

        // Calculate dL/dw
        tensor xt = matrix_transpose(x_i);
        tensor dw = matrix_multiply(dy_i, xt);
        tensor_axpy_(1, dw, l->dw);

        // Calculate dL/dx
        tensor col = matrix_multiply(wt, dy_i);
        tensor dx_i = col2im(col, x.size[1], x.size[2], x.size[3], f_h, f_w, l->stride, l->pad);
        memcpy(dx.data + i*tensor_len(dx_i), dx_i.data, tensor_len(dx_i) * sizeof(float));

        tensor_free(x_i);
        tensor_free(xt);
        tensor_free(dw);
        tensor_free(col);
        tensor_free(dx_i);
    }
    tensor_free(wt);
    tensor_free(w);
    return dx;
}

// Update convolutional layer
// layer l: layer to update
// float rate: learning rate
// float momentum: momentum term
// float decay: l2 regularization term
void update_convolutional_layer(layer *l, float rate, float momentum, float decay)
{
    tensor_axpy_(decay, l->w, l->dw);
    tensor_axpy_(-1 * rate, l->dw, l->w);
    tensor_scale_(rate * momentum, l->dw);
    tensor_axpy_(-1 * rate, l->db, l->b);
    tensor_scale_(rate * momentum, l->db);
}

// Make a new convolutional layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of convolutional filter to apply
// int stride: stride of operation

layer make_convolutional_layer(size_t c, size_t n, size_t size, size_t stride, size_t pad)
{
    layer l = {0};
    l.w  = tensor_vrandom(sqrtf(2.f/(c*size*size)), 4, n, c, size, size);
    l.dw = tensor_vmake(4, n, c, size, size);
    l.b  = tensor_vmake(1, n);
    l.db = tensor_vmake(1, n);
    l.stride = stride;
    l.pad = pad;
    l.size = size;
    l.forward  = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update   = update_convolutional_layer;
    return l;
}

