#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 8192
#define N 256
#define P 8192
#define MATRIX_ALIGN 64

double wall_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
  float *A = (float*)aligned_alloc(MATRIX_ALIGN, M*N*sizeof(float));
  float *B = (float*)aligned_alloc(MATRIX_ALIGN, N*P*sizeof(float));
  float *C = (float*)aligned_alloc(MATRIX_ALIGN, (size_t)M*P*sizeof(float));

  if (!A || !B || !C) {
    printf("Memory allocation failed\n");
    return 1;
  }

  for(int i=0; i<M*N; i++) { A[i] = (float)rand()/RAND_MAX; }
  for(int i=0; i<N*P; i++) { B[i] = (float)rand()/RAND_MAX; }
  
  for (int i=0; i<100000; i++) {
    double start = wall_time();
    vDSP_mmul(A, 1, B, 1, C, 1, M, P, N);
    double end = wall_time();

    double ops = 2.0 * N * M * P;
    double gflops = ops / (end - start) / 1e9;
    
    printf("Matrix size: %dx%d\n", M, P);
    printf("Time: %.3f s\n", (end - start)/10);
    printf("Performance: %.2f GFLOPS\n", gflops);
  }

  free(A);
  free(B);
  free(C);
  return 0;
}
