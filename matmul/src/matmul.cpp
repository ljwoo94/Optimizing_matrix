#include "matmul.h"
#include <omp.h>

int THREADS;

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
 // printMatrix(matrixC,n);
}

int* const mat_add(const int* const matrixA, const int* const matrixB, const int n) {
  int* const tmp = (int*)malloc(sizeof(int) * n * n);
  
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      tmp[i*n +j] = matrixA[i*n+j] + matrixB[i*n+j];

  return tmp;
}

int* const mat_sub(const int* const matrixA, const int* const matrixB, const int n) {
  int* const tmp = (int*)malloc(sizeof(int) * n * n);
  
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      tmp[i*n +j] = matrixA[i*n+j] - matrixB[i*n+j];

  return tmp;
}

void matmul_transpose(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n) {
  #pragma omp parallel for
  for(int i = 0; i < n; i++) {
    for(int k = 0; k < n; k++) {
      int matA = matrixA[i*n+k];
      for(int j = 0; j < n; j++)
        matrixC[i*n+j] += matA * matrixB[k*n+j];
    }
  }
}

void Strassen(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n) {
  
  //apply Strassen Algorithm
  //threashold of cache = 128;
  if(n <= 128) {
      matmul_transpose(matrixA, matrixB, matrixC, n);
      return;
  } else {

    int nn = n/2;
    int* const A11 = (int*)malloc(sizeof(int) * nn * nn); 
    int* const A12 = (int*)malloc(sizeof(int) * nn * nn); 
    int* const A21 = (int*)malloc(sizeof(int) * nn * nn); 
    int* const A22 = (int*)malloc(sizeof(int) * nn * nn); 
    int* const B11 = (int*)malloc(sizeof(int) * nn * nn); 
    int* const B12 = (int*)malloc(sizeof(int) * nn * nn); 
    int* const B21 = (int*)malloc(sizeof(int) * nn * nn); 
    int* const B22 = (int*)malloc(sizeof(int) * nn * nn);

    int i,j,k;
    for(i = 0; i < nn; i++) {
      for(j = 0; j < nn; j++) {
        A11[i*nn + j] = matrixA[i * n + j];
        A12[i*nn + j] = matrixA[i * n + j + nn];
        A21[i*nn + j] = matrixA[(i+nn)*n + j];
        A22[i*nn + j] = matrixA[(i+nn)*n + j + nn];
        B11[i*nn + j] = matrixB[i * n + j];
        B12[i*nn + j] = matrixB[i * n + j + nn];
        B21[i*nn + j] = matrixB[(i+nn)*n + j];
        B22[i*nn + j] = matrixB[(i+nn)*n + j + nn];
      }
    }
    
    int*  M1 = (int*)calloc(nn*nn, sizeof(int));  
    int*  M2 = (int*)calloc(nn*nn, sizeof(int));
    int*  M3 = (int*)calloc(nn*nn, sizeof(int));
    int*  M4 = (int*)calloc(nn*nn, sizeof(int)); 
    int*  M5 = (int*)calloc(nn*nn, sizeof(int));
    int*  M6 = (int*)calloc(nn*nn, sizeof(int));
    int*  M7 = (int*)calloc(nn*nn, sizeof(int));
    
    //int *P1, *P2, *P3, *P4, *P5, *P6, *P7,*P8, *P9, *P10;

    //need to make strassens parallel.
    #pragma omp parallel sections
    {
      #pragma omp section
        Strassen(mat_add(A11,A22,nn), mat_add(B11,B22,nn), M1, nn);
      #pragma omp section
        Strassen(mat_add(A21,A22,nn), B11, M2, nn);
      #pragma omp section
        Strassen(A11, mat_sub(B12,B22,nn), M3, nn);
      #pragma omp section
       Strassen(A22, mat_sub(B21,B11,nn), M4, nn);
      #pragma omp section
        Strassen(mat_add(A11,A12,nn), B22, M5, nn);
      #pragma omp section
        Strassen(mat_sub(A21,A11,nn), mat_add(B11,B12,nn), M6, nn);
      #pragma omp section
        Strassen(mat_sub(A12,A22,nn), mat_add(B21,B22,nn), M7, nn);
    }

    int*  C11 = (int*)malloc(sizeof(int) * nn * nn); 
    int*  C12 = (int*)malloc(sizeof(int) * nn * nn); 
    int*  C21 = (int*)malloc(sizeof(int) * nn * nn); 
    int*  C22 = (int*)malloc(sizeof(int) * nn * nn); 

  for(i=0; i<nn*nn; i++) {
    C11[i] = M1[i] + M4[i] - M5[i] + M7[i];
    C12[i] = M3[i] + M5[i];
    C21[i] = M2[i] + M4[i];
    C22[i] = M1[i] - M2[i] + M3[i] + M6[i];
  }
    //merge C
    for(i = 0; i < nn; i++) {
      for(j = 0; j < nn; j++) {
        matrixC[i*n+j] = C11[i*nn+j];
        matrixC[i*n+j+nn] = C12[i*nn+j];
        matrixC[(i+nn)*n+j] = C21[i*nn+j];
        matrixC[(i+nn)*n+j+nn] = C22[i*nn+j];
      }
    }

      free(A11);free(A12);free(A21);free(A22);
      free(B11);free(B12);free(B21);free(B22);
      free(C11);free(C12);free(C21);free(C22);
      free(M1);free(M2);free(M3);free(M4);free(M5);free(M6);free(M7);
      //free(P1);free(P2);free(P3);free(P4);free(P5);free(P6);free(P7);free(P8);free(P9);free(P10);
      return;
    }
  }

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n) {
    THREADS = omp_get_num_procs();
    omp_set_num_threads(THREADS);
    Strassen(matrixA,matrixB,matrixC,n);
    //printMatrix(matrixC,n);
}
