#include "matmul.h"

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

  int b = 52;
  for(int i = b; i > 0; i++) {
    if(n%i == 0) {
      b = i;
      break;
    }
  }

  //block + trans single thread
  for(int i=0; i<n; i+=b)
    for(int k=0; k<n; k+=b)
      for(int j=0; j<n; j+=b)   
        for(int ii=0; ii<b; ii++)
          for(int kk=0; kk<b; kk++)
            for(int jj=0; jj<b; jj++)
              matrixC[(i+ii)*n + j+jj] += matrixA[(i+ii)*n + k+kk]*matrixB[(k+kk)*n + j+jj];
}
