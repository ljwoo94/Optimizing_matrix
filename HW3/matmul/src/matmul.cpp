#include "matmul.h"
#include <thread>

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n) {
  // TODO: Implement your code

  //Blocked Matrix Multiply L1 cache = 32K
  //int = 4byte. 32K = 32,768 B
  //L1d can hold 8192 ints.
  //cache line = 64 bit = 8 B => 2 consecutive ints
  //about 52x52 matrix = 1 L1cache without capacity miss
  int b = 52;
  for(int i = b; i > 0; i++) {
    if(n%i == 0) {
      b = i;
      break;
    }
  }
  omp_set_num_threads(omp_get_num_procs());
  #pragma omp parallel for
   for(int i=0; i<n; i+=b)
        for(int j=0; j<n; j+=b)
            for(int k=0; k<n; k+=b) {

              #pragma omp parallel for
               for(int ii=0; ii<b; ii++)
                    for(int jj=0; jj<b; jj++)
                        for(int kk=0; kk<b; kk++)
                            matrixC[(i+ii)*n + j+jj] += matrixA[(i+ii)*n + k+kk]*matrixB[(k+kk)*n + j+jj];
            }
}