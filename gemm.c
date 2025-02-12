// gcc -O3 gemm.c
// gcc gemm.c -O2 -ffast-math && ./a.out
// CPU Freq: 3.2GHz

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <arm_neon.h>
#include <stdbool.h>

// #define N 256
#define M 4096
#define N 1024
#define P 2048
#define BYTE_BOUNDARY 16
#define BLOCK_I 128
#define BLOCK_J 32
#define N_LANES 4
#define WORD_SIZE 4

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
void pretty_print(float *mat, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%8.3f ", mat[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// this implementation is right. Checked it with numpy
void matmul(float *a, float *b, float *c) {
  int BLOCK_SIZE = 128;
  double start = get_time();
  float *tmp = (float *)aligned_alloc(BYTE_BOUNDARY, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

  for (int bi=0; bi<M; bi+=BLOCK_SIZE) {
    for (int bj=0; bj<P; bj+=BLOCK_SIZE) {

      // compute
      memset(tmp, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
      for (int k=0; k<N; k++) {
        for (int i=0; i<BLOCK_SIZE; i++) {
          for (int j=0; j<BLOCK_SIZE; j++) { 
            tmp[i*BLOCK_SIZE+j] += a[(bi+i)*N+k] * b[(k)*N + bj+j];
          }
        }

      }

      // store
      for (int i=0; i<BLOCK_SIZE; i++) {
        for (int j=0; j<BLOCK_SIZE; j++) { 
          c[(bi+i)*P+(bj+j)] = tmp[i*BLOCK_SIZE+j];
        }
      }
    }
  }
  double end = get_time();
  float time_spent = (float) (end - start);
  float flop = N*N*(2.0*N);
  float flop_s = flop / time_spent;
  printf("Correct Matmul : %f GFLOP/s\n", flop_s/(1e9));
}



void fast_matmul(float *a, float *b, float*c) {
  memset(c, 0, M*P*sizeof(float));

  float32x4_t* b_swizzled = (float32x4_t*)aligned_alloc(BYTE_BOUNDARY, (N/N_LANES) * (BLOCK_J*N_LANES/WORD_SIZE) * sizeof(float32x4_t));
  float *tmp = (float *)aligned_alloc(BYTE_BOUNDARY, BLOCK_I * BLOCK_J * sizeof(float));

  for (int bi=0; bi<M; bi+=BLOCK_I) {
    for (int bj=0; bj<P; bj+=BLOCK_J) {

      // load (pre-swizzle)
      for (int k=0; k<N; k+=N_LANES) {
        for (int j=0; j<BLOCK_J; j+=WORD_SIZE) {
          for (int l=0; l<N_LANES; l++) {
            int row = (k/N_LANES)*(BLOCK_J*N_LANES/WORD_SIZE);
            int col = (j/WORD_SIZE) * N_LANES + l;
            b_swizzled[row+col] = vld1q_f32(&b[(k+l)*N + (bj+j)]);
          }
        }
      }

      // compute
      memset(tmp, 0, BLOCK_I * BLOCK_J * sizeof(float));
      for (int k=0; k<N; k+=N_LANES) {

        for (int i=0; i<BLOCK_I; i++) {
          float32x4_t a_vec = vld1q_f32(&a[(bi+i)*N+k]);
          int row = (k/N_LANES)*(BLOCK_J*N_LANES/WORD_SIZE);
          for (int j=0; j<BLOCK_J; j+=WORD_SIZE) {
            float32x4_t acc = vld1q_f32(&tmp[i*BLOCK_J+j]);
            int col = (j/WORD_SIZE) * N_LANES;
            acc = vfmaq_laneq_f32(acc, b_swizzled[row+col+0], a_vec, 0);
            acc = vfmaq_laneq_f32(acc, b_swizzled[row+col+1], a_vec, 1);
            acc = vfmaq_laneq_f32(acc, b_swizzled[row+col+2], a_vec, 2);
            acc = vfmaq_laneq_f32(acc, b_swizzled[row+col+3], a_vec, 3);
            vst1q_f32(&tmp[i * BLOCK_J + j], acc);
          }
        }
      }

      // store
      for (int i=0; i<BLOCK_I; i++) { for (int j=0; j<BLOCK_J; j++) c[(bi+i)*P+bj+j] = tmp[i*BLOCK_J+j]; }
    }
  }
  free(tmp);
  free(b_swizzled);
}

int main() {
  // The main reason for the segfault is likely that stack-allocated arrays (your first approach) are limited by stack size, which is typically much smaller than heap space. When N*N is large, you're likely exceeding the stack size limit, causing a stack overflow and segfault.
  // float a[N*N] __attribute__((aligned(BYTE_BOUNDARY)));

  // Align memory to 16-byte boundary for better cache performance
  // The heap-allocated version (using aligned_alloc) works because:
  // The heap has much more available space
  // aligned_alloc properly handles the alignment requirements
  // 16-byte alignment becomes noticeable because it matches SIMD register sizes (like SSE's 128-bit registers)
  float *a = (float *)aligned_alloc(BYTE_BOUNDARY, M * N * sizeof(float));
  float *b = (float *)aligned_alloc(BYTE_BOUNDARY, N * P * sizeof(float));
  float *c = (float *)aligned_alloc(BYTE_BOUNDARY, M * P * sizeof(float));
  float *cc = (float *)aligned_alloc(BYTE_BOUNDARY, M * P * sizeof(float));

  // Initialize matrices with random values
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a[i * N + j] = (float)rand() / RAND_MAX;
      // a[i * N + j] = i * N + j;
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < P; j++) {
      b[i * P + j] = (float)rand() / RAND_MAX;
      // b[i * P + j] = i * P + j;
    }
  }
  memset(cc, 0, M*P*sizeof(float));

  for (int i=0; i<10; i++){
    double start = get_time();

    fast_matmul(a, b, c);

    double end = get_time();
    float time_spent = (float) (end - start);
    float flop = M*P*(2.0*N);
    float flop_s = flop / time_spent;
    printf("   Fast Matmul : %f GFLOP/s\n", flop_s/(1e9));
  }


  matmul(a, b, cc);
  bool is_correct = true;
  for (int i=0; i<(N*N); i++) {
    if (fabsf(cc[i] - c[i]) > 1e-4) {
      printf("MISMATCH at [%d]: %f != %f\n", i, cc[i], c[i]);
      is_correct = false;
      break;
    }
  }
  if (is_correct) printf("MATCHED!!!!\n");

  // printf("\nMatrix C (Fast):\n");
  // pretty_print(c, N);
  // printf("\nMatrix CC (Reference):\n"); 
  // pretty_print(cc, N);

  free(a);
  free(b);
  free(c);
  free(cc);

  return 0;
}
