#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "dubnet.h"

float math_max(float a, float b) {
    return (a <= b) ? b : a;
}

// float a: the old value to compare to
// float b: the new value to compare to a
// returns if b is a new maximum value (greater than or equal to)
// returns TRUE if b is greater or equal to a
// returns FALSE otherwise
int is_new_max(float old, float new) {
    return old <= new;
}

// THIS IS THE REAL ONE TO USE
size_t get_max(
        tensor x,
        size_t start,
        size_t len,
        size_t kernel_width,
        size_t img_height,
        size_t img_width,
        size_t x_start,
        size_t y_start,
        int pad) {
    int is_set = 0;
    float max;
    size_t max_pos;

    size_t i;
    for (i = 0; i < len; ++i) {
        size_t x_pos = (i % kernel_width) + x_start;
        size_t y_pos = (i / kernel_width) + y_start;
//        int invalidX = x_pos + pad < 0 || x_pos + pad >= img_width;
//        int invalidY = y_pos + pad < 0 || y_pos + pad >= img_height;
        int invalidX;
        int invalidY;
        if (pad != 0) {
            invalidX = x_pos < (-1) * pad || x_pos + pad >= img_width;
            invalidY = y_pos < (-1) * pad || y_pos + pad >= img_height;
        } else {
            invalidX = x_pos < 0 || x_pos >= img_width;
            invalidY = y_pos < 0 || y_pos >= img_height;
        }
        // We only need to do something if the pos is valid
        if (!invalidX && !invalidY) {
            size_t pos = ((y_pos + pad) * img_width) + (x_pos + pad) + start;
            float curr = x.data[pos];
            if (!is_set || is_new_max(max, curr)) {
                max = curr;
                max_pos = pos;
                is_set = 1;
            }
        }
    }

    return max_pos;
}

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
tensor forward_maxpool_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    assert(x.n == 4);

    size_t res_h = (x.size[2]-1)/l->stride + 1;
    size_t res_w = (x.size[3]-1)/l->stride + 1;

    tensor y = tensor_vmake(4,
        x.size[0],  // same # data points and # of channels (N and C)
        x.size[1],
        res_h, // H and W scaled based on stride
        res_w);

    size_t img_h = x.size[2];
    size_t img_w = x.size[3];

    size_t IMAGE_AREA = img_h * img_w;
    size_t NUM_IMAGES = x.size[0];
    size_t NUM_CHANNELS = x.size[1];
    // the area for all the channels within an image
    size_t IMAGE_X_CHANNEL_AREA = NUM_CHANNELS * IMAGE_AREA;
    // the area for the result image
    size_t RES_AREA = res_h * res_w;
    size_t RES_X_CHANNEL_AREA = NUM_CHANNELS * RES_AREA;

    // This might be a useful offset...
    int pad = -((int) l->size - 1)/2;

    // TEST CODE START
//    size_t test_kernel_h = 3;
//    size_t test_kernel_w = 3;
//    size_t test_img_h = 5;
//    size_t test_img_w = 5;
//    size_t test_stride = 2;
//    int test_pad = -((int) test_kernel_w - 1)/2;
//
//    size_t test_res_h = (test_img_h - 1)/test_stride + 1;
//    size_t test_res_w = (test_img_w - 1)/test_stride + 1;

    // TEST CODE END

//    tensor_simple_print(x);
//    tensor_simple_print(y);
//    printf("padding: %d \n\n", pad);
//    printf("res_h: %zu, res_w: %zu, pad: %d, size: %zu, stride: %zu\n\n", test_res_h, test_res_w, test_pad, test_kernel_h, test_stride);
//    printf("res_h: %zu, res_w: %zu \n\n", res_h, res_w);
//    tensor_simple_print(y);
    size_t image, channel;

    for (image = 0; image < NUM_IMAGES; ++image) {
        for (channel = 0; channel < NUM_CHANNELS; ++channel) {
            size_t i;
            for (i = 0; i < res_h; ++i) {
                size_t j;
                for (j = 0; j < res_w; ++j) {
                    // these start positions are relative to the image only
                    size_t x_start = j * l->stride;
                    size_t y_start = i * l->stride;
                    size_t curr_max_pos = get_max(x,
                                             (image * IMAGE_X_CHANNEL_AREA) + (channel * IMAGE_AREA),
                                             l->size * l->size,
                                             l->size,
                                             img_h,
                                             img_w,
                                             x_start,
                                             y_start,
                                             pad);
                    size_t output_x_pos = j;
                    size_t output_y_pos = i;
                    size_t output_pos = (output_y_pos * res_w) +
                            output_x_pos +
                            (channel * RES_AREA) +
                            (image * RES_X_CHANNEL_AREA);
                    y.data[output_pos] = x.data[curr_max_pos];
                    // apparently 289 is an issue
                }
            }
        }
    }
    return y;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
tensor backward_maxpool_layer(layer *l, tensor dy)
{
    tensor x    = l->x;
    tensor dx = tensor_make(x.n, x.size);
    int pad = -((int) l->size - 1)/2;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    // TODO: I think we find the pos of the max value given the layer
    // TODO: Then we fill in that position with the delta of the same position found in dy

    size_t res_h = (x.size[2]-1)/l->stride + 1;
    size_t res_w = (x.size[3]-1)/l->stride + 1;

    size_t img_h = x.size[2];
    size_t img_w = x.size[3];

    size_t IMAGE_AREA = img_h * img_w;
    size_t NUM_IMAGES = x.size[0];
    size_t NUM_CHANNELS = x.size[1];
    // the area for all the channels within an image
    size_t IMAGE_X_CHANNEL_AREA = NUM_CHANNELS * IMAGE_AREA;
    // the area for the result image
    size_t RES_AREA = res_h * res_w;
    size_t RES_X_CHANNEL_AREA = NUM_CHANNELS * RES_AREA;
    // the area for the kernel
    size_t KERN_AREA = l->size * l->size;

    size_t image, channel;

//    printf("result area: %zu \n", RES_AREA);
//    printf("layer size: %zu \n", l->size);
//    printf("img h: %zu \n", img_h);
//    printf("img w: %zu \n", img_w);

    for (image = 0; image < NUM_IMAGES; ++image) {
        for (channel = 0; channel < NUM_CHANNELS; ++channel) {
            size_t i;
            for (i = 0; i < res_h; ++i) {
                size_t j;
                for (j = 0; j < res_w; ++j) {
                    // these start positions are relative to the image only
                    size_t x_start = j * l->stride;
                    size_t y_start = i * l->stride;
                    size_t max_pos = get_max(x,
                                             (image * IMAGE_X_CHANNEL_AREA) + (channel * IMAGE_AREA),
                                             KERN_AREA,
                                             l->size,
                                             img_h,
                                             img_w,
                                             x_start,
                                             y_start,
                                             pad);
                    size_t output_x_pos = j;
                    size_t output_y_pos = i;
                    size_t output_pos = (output_y_pos * res_w) +
                                        output_x_pos +
                                        (channel * RES_AREA) +
                                        (image * RES_X_CHANNEL_AREA);
                    dx.data[max_pos] += dy.data[output_pos];
                }
            }
        }
    }
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer *l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(size_t size, size_t stride)
{
    layer l = {0};
    l.size = size;
    l.stride = stride;
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

