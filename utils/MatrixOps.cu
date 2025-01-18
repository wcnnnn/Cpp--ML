#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <vector>
using namespace std;

class MatrixOps
{
public:
    static __global__ void matrix_matrix_multiply(
        double* x, double* y, double* z, int m, int n, int p){
        int i = blockIdx.y * blockDim.y+ threadIdx.y;
        

    };
   
};
