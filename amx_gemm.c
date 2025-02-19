#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 1024
#define N 256
#define P 1024
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
  
  for (int i=0; i<10; i++) {
    double start = wall_time();
    vDSP_mmul(A, 1, B, 1, C, 1, M, P, N);
    double end = wall_time();

    double ops = 2.0 * N * M * P;
    double gflops = ops / (end - start) / 1e9;
    
    printf("Matrix size: %dx%d\n", M, P);
    printf("Time: %.3f s\n", (end - start)/10);
    printf("Performance: %.2f GFLOPS\n", gflops);
  }


  // test output by doing normal 3 loop matmul
  float *C_test = (float*)aligned_alloc(MATRIX_ALIGN, (size_t)M*P*sizeof(float));
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < P; j++) {
      C_test[i*P + j] = 0;
      for(int k = 0; k < N; k++) {
        C_test[i*P + j] += A[i*N + k] * B[k*P + j];
      }
    }
  }

  // Compare results
  float max_diff = 0.0f;
  for(int i = 0; i < M*P; i++) {
    float diff = fabsf(C[i] - C_test[i]);
    if(diff > max_diff) max_diff = diff;
  }
  printf("Maximum difference between BLAS and naive implementation: %e\n", max_diff);

  free(A);
  free(B);
  free(C);
  free(C_test);

  return 0;
}
