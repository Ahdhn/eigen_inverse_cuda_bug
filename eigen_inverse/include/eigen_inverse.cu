#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "helper.h"

#include <Eigen/Dense>

__global__ void foo()
{
    Eigen::Matrix2f M;
    M << 10, 2,  //
        4, 10;

    printf("\n M = \n {%f, %f \n %f, %f}", M(0, 0), M(0, 1), M(1, 0), M(1, 1));

    auto M_inv = M.inverse();

    printf("\n M_inv = \n {%f, %f \n %f, %f}",
           M_inv(0, 0),
           M_inv(0, 1),
           M_inv(1, 0),
           M_inv(1, 1));
}

int main(int argc, char** argv)
{
    foo<<<1, 1>>>();

    CUDA_ERROR(cudaDeviceSynchronize());
    return 0;
}
