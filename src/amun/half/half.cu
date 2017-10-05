#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>
#include <curand.h>
#include <cuda_fp16.h>
#include <chrono>

/////////////////////////////////////////////////////////////////////////////

void GPU_fill_rand(half *A, int nr_rows_A, int nr_cols_A) {
     // Create a pseudo-random number generator
     curandGenerator_t prng;
     curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

     // Set the seed for the random number generator using the system clock
     curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

     // Fill the array with random numbers on the device
     /* curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A); */
}

/////////////////////////////////////////////////////////////////////////////

void gpu_blas_mmul(const half *A, const half *B, half *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;

     half alf_h;
     half *alpha_h = &alf_h;

     half bet_h;
     half *beta_h = &bet_h;

     // Create a handle for CUBLAS
     cublasHandle_t handle;
     cublasCreate(&handle);


     // Do the actual multiplication
     for (size_t i = 0; i < 1; ++i) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_h,
            A, lda, B, ldb, beta_h, C, ldc);
     }
     cudaStreamSynchronize(0);

     // Destroy the handle
     cublasDestroy(handle);
}

/////////////////////////////////////////////////////////////////////////////

__global__ void gPlusTanh(const half *A, const half *B, half *C, size_t size)
{
  int i = threadIdx.x  + blockDim.x * blockIdx.x;
  if (i < size) {
    half res = A[i] + B[i];
    //res = tanh(res);
    C[i] = res; 
  }
}

/////////////////////////////////////////////////////////////////////////////

int main() {
    std::chrono::time_point<std::chrono::system_clock> start, end1, end2;
    start = std::chrono::system_clock::now();

     // Allocate 3 arrays on CPU
     int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

     // for simplicity we are going to use square arrays
     nr_rows_A = 512;
     nr_cols_A = 512;
     nr_rows_B = 512;
     nr_cols_B = 85000;
     nr_rows_C = 520;
     nr_cols_C = 85000;

     // Allocate 3 arrays on GPU
     half *d_A, *d_B, *d_C;
     cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(half));
     cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(half));
     cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(half));

     for (size_t i = 0; i < 1000; ++i) {
		 // Fill the arrays A and B on GPU with random numbers
		 GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
		 GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

		 // Multiply A and B on GPU
		 gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
     }

     // Copy (and print) the result on host memory

     //Free GPU memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);  

     end1 = std::chrono::system_clock::now();
     std::chrono::duration<double> elapsed1 = end1 - start;
     std::cout << "multiplication: " << elapsed1.count() << "s\n";

     // element-wise tanh(x+y)
     nr_rows_A = 520;
     nr_cols_A = 85000;
     nr_rows_B = 520;
     nr_cols_B = 85000;
     int size = nr_rows_A * nr_cols_A;

     cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
     cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
     cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

     size_t threads = 512;
     size_t blocks =  (size / threads) + ((size % threads == 0) ?  0 : 1);

     GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
     GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

     for (size_t i = 0; i < 1000; ++i) {
       gPlusTanh<<<blocks, threads>>>(d_A, d_B, d_C, size);
     }
     cudaStreamSynchronize(0);

     //Free GPU memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);  

     end2 = std::chrono::system_clock::now();
     std::chrono::duration<double> elapsed2 = end2 - end1;
     std::cout << "element-wise tanh(x+y): " << elapsed2.count() << "s\n";

     return 0;
 }
