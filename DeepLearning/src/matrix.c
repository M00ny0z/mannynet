#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"

// Transpose a matrix
// tensor m: matrix to be transposed
// returns: tensor, result of transposition
// DONE
tensor matrix_transpose(tensor a)
{
    assert(a.n == 2);
    size_t new_size[2] = {a.size[1], a.size[0]};
    tensor t = tensor_make(2, new_size);

    // original amount of rows
    size_t rows = a.size[0];
    // original amount of columns
    size_t cols = a.size[1];
    size_t i;
    for (i = 0; i < cols; ++i) {
        size_t j;
        for (j = 0; j < rows; ++j) {
            t.data[j + (i * rows)] = a.data[(j * cols) + i];
        }
    }

    return t;
}

// Perform matrix multiplication a*b, return result
// tensor a,b: operands
// returns: new tensor that is the result
// BELOW IS THE TRANSPOSE IMPLEMENTATION
tensor matrix_multiply(const tensor a, const tensor b)
{
    assert(a.n == 2);
    assert(b.n == 2);
    assert(a.size[1] == b.size[0]);
    // END MATRIX WILL BE OF FORM (a_rows X b_cols)
    size_t t_size[2] = { a.size[0], b.size[1]};
    tensor t = tensor_make(2, t_size);
    tensor b_t = matrix_transpose(b);

    size_t a_rows = a.size[0];
    size_t a_cols = a.size[1];
    size_t b_rows = b_t.size[0];
    size_t b_cols = b_t.size[1];

    // i goes through every row in A
    size_t i;
    for (i = 0; i < a_rows; ++i) {
        // j goes through every row in B_T
        size_t j;
        for (j = 0; j < b_rows; ++j) {
            // k goes through every number inside a row in B
            size_t k;
            float sum = 0;
            for (k = 0; k < b_cols; ++k) {
                sum += b_t.data[(j * b_cols) + k] * a.data[(i * a_cols) + k];
            }
            t.data[(i * b_rows) + j] = sum;
        }
    }
    tensor_free(b_t);
    return t;
}

// Used for matrix inversion
tensor matrix_augment(tensor m)
{
    assert(m.n == 2);
    size_t rows = m.size[0];
    size_t cols = m.size[1];
    size_t i,j;
    tensor c = tensor_vmake(2, rows, cols*2);
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            c.data[i*cols*2 + j] = m.data[i*cols + j];
        }
    }
    for(j = 0; j < rows; ++j){
        c.data[j*cols*2 + j+cols] = 1;
    }
    return c;
}

// Invert matrix m
tensor matrix_invert(tensor m)
{
    size_t i, j, k;
    assert(m.n == 2);
    assert(m.size[0] == m.size[1]);

    tensor c = matrix_augment(m);
    tensor none = {0};
    float **cdata = calloc(c.size[0], sizeof(float *));
    for(i = 0; i < c.size[0]; ++i){
        cdata[i] = c.data + i*c.size[1];
    }

    for(k = 0; k < c.size[0]; ++k){
        float p = 0.;
        size_t index = -1;
        for(i = k; i < c.size[0]; ++i){
            float val = fabs(cdata[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Can't do it, sorry!\n");
            tensor_free(c);
            return none;
        }

        float *swap = cdata[index];
        cdata[index] = cdata[k];
        cdata[k] = swap;

        float val = cdata[k][k];
        cdata[k][k] = 1;
        for(j = k+1; j < c.size[1]; ++j){
            cdata[k][j] /= val;
        }
        for(i = k+1; i < c.size[0]; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.size[1]; ++j){
                cdata[i][j] +=  s*cdata[k][j];
            }
        }
    }
    for(k = c.size[0]-1; k > 0; --k){
        for(i = 0; i < k; ++i){
            float s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < c.size[1]; ++j){
                cdata[i][j] += s*cdata[k][j];
            }
        }
    }
    tensor inv = tensor_make(2, m.size);
    for(i = 0; i < m.size[0]; ++i){
        for(j = 0; j < m.size[1]; ++j){
            inv.data[i*m.size[1] + j] = cdata[i][j+m.size[1]];
        }
    }
    tensor_free(c);
    free(cdata);
    return inv;
}

tensor solve_system(tensor M, tensor b)
{
    tensor none = {0};
    tensor Mt = matrix_transpose(M);
    tensor MtM = matrix_multiply(Mt, M);
    tensor MtMinv = matrix_invert(MtM);
    if(!MtMinv.data) return none;
    tensor Mdag = matrix_multiply(MtMinv, Mt);
    tensor a = matrix_multiply(Mdag, b);
    tensor_free(Mt);
    tensor_free(MtM);
    tensor_free(MtMinv);
    tensor_free(Mdag);
    return a;
}
