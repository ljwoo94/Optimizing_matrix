#include "matmul.h"
#include <omp.h>
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

  int b = 32;
  for(int i = b; i > 0; i++) {
    if(n%i == 0) {
      b = i;
      break;
    }
  }
  omp_set_num_threads(omp_get_num_procs());
  //block + trans multi-thread
  #pragma omp parallel for
  for(int i=0; i<n; i+=b)
    for(int k=0; k<n; k+=b)
     // #pragma omp parallel for 
      for(int j=0; j<n; j+=b)
    //  #pragma omp parallel for   
        for(int ii=0; ii<b; ii++)
          for(int kk=0; kk<b; kk++) {
            int matA = matrixA[(i+ii)*n+ k+kk];
             for(int jj=0; jj<b; jj++)
              matrixC[(i+ii)*n + j+jj] += matA*matrixB[(k+kk)*n + j+jj];
          }    
}
