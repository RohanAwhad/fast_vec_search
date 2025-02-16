// clang -O3 amx_gemm.c -DACCELERATE_NEW_LAPACK -march=native  -framework Accelerate

#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 4096
#define MATRIX_ALIGN 64

double wall_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
  float *A = (float*)aligned_alloc(MATRIX_ALIGN, N*N*sizeof(float));
  float *B = (float*)aligned_alloc(MATRIX_ALIGN, N*N*sizeof(float));
  float *C = (float*)aligned_alloc(MATRIX_ALIGN, N*N*sizeof(float));

  for(int i=0; i<N*N; i++) {
    A[i] = (float)rand()/RAND_MAX;
    B[i] = (float)rand()/RAND_MAX;
  }

  const enum CBLAS_ORDER Order = CblasRowMajor;
  const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
  const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
  
  const int M = N;
  const int K = N; 
  const int LDA = N;
  const int LDB = N;
  const int LDC = N;
  
  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int i=0; i<1000; i++) {
    double start = wall_time();
    cblas_sgemm(Order, TransA, TransB, M, N, K,
                alpha, A, LDA, B, LDB, beta, C, LDC);

    double end = wall_time();

    double ops = 2.0 * N * N * N;
    double gflops = ops / (end - start) / 1e9;
    
    printf("Matrix size: %dx%d\n", N, N);
    printf("Time: %.3f s\n", (end - start)/10);
    printf("Performance: %.2f GFLOPS\n", gflops);
  }

  free(A);
  free(B);
  free(C);
  return 0;
}
