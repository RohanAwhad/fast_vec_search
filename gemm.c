#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
// #include <arm_neon.h>

#define N 2048
#define BLOCK_SIZE 16

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

// this implementation is right. Checked it with numpy
void matmul(float *a, float *b, float *c) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) { 
      for (int k=0; k<N; k++) {
        c[i*N+j] += a[i*N+k] * b[k*N+j]; 
      }
    }
  }
}

void fast_matmul(float *a, float *b, float*c) {
  double start = get_time();
  // Matrix multiplication
  for (int i=0; i<(N*N); i++) c[i] = 0;

  float *tmp = (float *)malloc(N * N * sizeof(float));
  // float32x4_t a;
  // float32x4_t b;
  // float32x4_t c;

  for (int bi=0; bi<N; bi+=BLOCK_SIZE) {
    for (int bj=0; bj<N; bj+=BLOCK_SIZE) {

      
      // compute
      for (int l=0; l<(BLOCK_SIZE*BLOCK_SIZE); l++) tmp[l] = 0;

      for (int bk=0; bk<N; bk+=BLOCK_SIZE) {

        for (int i=0; i<BLOCK_SIZE; i++) {
          for (int j=0; j<BLOCK_SIZE; j++) {

            float acc = 0;
            for (int k=0; k<BLOCK_SIZE; k++) {
              acc += a[(bi+i)*N+(bk+k)] * b[(bk+k)*N+(bj+j)];
            }


            tmp[i*N+j] += acc;
          }
        }

      }

      // store
      for (int i=0; i<BLOCK_SIZE; i++) {
        for (int j=0; j<BLOCK_SIZE; j++) {
          c[(bi+i)*N+bj+j] = tmp[i*BLOCK_SIZE+j];
        }
      }


    }
  }
  free(tmp);
  double end = get_time();
  float time_spent = (float) (end - start);
  float flop = N*N*(2.0*N);
  float flop_s = flop / time_spent;
  printf("C : %f GFLOP/s\n", flop_s/(1e9));
}

int main() {
  float *a = (float *)malloc(N * N * sizeof(float));
  float *b = (float *)malloc(N * N * sizeof(float));
  float *c = (float *)malloc(N * N * sizeof(float));
  float *cc = (float *)malloc(N * N * sizeof(float));

  // Align memory to 16-byte boundary for better cache performance
  // float *a = (float *)aligned_alloc(16, N * N * sizeof(float));
  // float *b = (float *)aligned_alloc(16, N * N * sizeof(float));
  // float *c = (float *)aligned_alloc(16, N * N * sizeof(float));
  // float *cc = (float *)aligned_alloc(16, N * N * sizeof(float));

  // Initialize matrices with random values
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i * N + j] = (float)rand() / RAND_MAX;
      b[i * N + j] = (float)rand() / RAND_MAX;
      // c[i * N + j] = 0;
      cc[i * N + j] = 0;
    }
  }
  
  for (int i =0; i<100; i++) fast_matmul(a, b, c);

  matmul(a, b, cc);
  for (int i=0; i<(N*N); i++) {
    if (fabsf(cc[i] - c[i]) > 1e-5) {
      printf("MISMATCH at [%d]: %f != %f\n", i, cc[i], c[i]);
      return 0;
    }
  }

  free(a);
  free(b);
  free(c);

  return 0;
}
