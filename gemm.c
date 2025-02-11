// gcc -O3 gemm.c

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <arm_neon.h>

// #define N 256
#define N 2048
#define BYTE_BOUNDARY 16

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

// this implementation is right. Checked it with numpy
void matmul(float *a, float *b, float *c) {
  int BLOCK_SIZE = 128;
  double start = get_time();
  float *tmp = (float *)aligned_alloc(BYTE_BOUNDARY, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

  for (int bi=0; bi<N; bi+=BLOCK_SIZE) {
    for (int bj=0; bj<N; bj+=BLOCK_SIZE) {

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
          c[(bi+i)*N+(bj+j)] = tmp[i*BLOCK_SIZE+j];
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


// Helper function for transpose
inline int min(int a, int b) {
    return (a < b) ? a : b;
}

// could possibly use arm_neon intrinsics
void transpose(float *in, float *out, int n) {
    int BLOCK_SIZE = 8;
    for (int i = 0; i < n; i += BLOCK_SIZE*2) {
        for (int j = 0; j < n; j += BLOCK_SIZE*2) {
            // Transpose BLOCK_SIZE*2 x BLOCK_SIZE*2 blocks to improve cache usage
            for (int ib = i; ib < min(i + BLOCK_SIZE*2, n); ib++) {
                for (int jb = j; jb < min(j + BLOCK_SIZE*2, n); jb++) {
                    out[jb * n + ib] = in[ib * n + jb];
                }
            }
        }
    }
}

void fast_matmul(float *a, float *b, float*c) {
  int BLOCK_SIZE = 8;
  float *b_trans = (float *)aligned_alloc(BYTE_BOUNDARY, N * N * sizeof(float));
  transpose(b, b_trans, N);
  // i dont want to worry about transpose speed right now
  double start = get_time();
  memset(c, 0, N*N*sizeof(float));

  float *tmp = (float *)aligned_alloc(BYTE_BOUNDARY, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
  for (int bi=0; bi<N; bi+=BLOCK_SIZE) {
    for (int bj=0; bj<N; bj+=BLOCK_SIZE) {

      // compute
      memset(tmp, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
      for (int bk=0; bk<N; bk+=BLOCK_SIZE) {

        for (int i=0; i<BLOCK_SIZE; i++) {
          for (int j=0; j<BLOCK_SIZE; j++) {

            float32x4_t acc = (float32x4_t){0.0f, 0.0f, 0.0f, 0.0f};
            for (int k=0; k<BLOCK_SIZE; k+=4) {
              float32x4_t a_vec = vld1q_f32(&a[(bi + i)*N + bk+k]);
              float32x4_t b_vec = vld1q_f32(&b_trans[(bj + j)*N + bk+k]);
              acc = vfmaq_f32(acc, a_vec, b_vec);
            }
            tmp[i*BLOCK_SIZE + j] += vaddvq_f32(acc); // Sum all four elements

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
  printf("   Fast Matmul : %f GFLOP/s\n", flop_s/(1e9));
  free(b_trans);
}

int main() {
  // The main reason for the segfault is likely that stack-allocated arrays (your first approach) are limited by stack size, which is typically much smaller than heap space. When N*N is large, you're likely exceeding the stack size limit, causing a stack overflow and segfault.
  // float a[N*N] __attribute__((aligned(BYTE_BOUNDARY)));
  // float b[N*N] __attribute__((aligned(BYTE_BOUNDARY)));
  // float c[N*N] __attribute__((aligned(BYTE_BOUNDARY)));
  // float cc[N*N] __attribute__((aligned(BYTE_BOUNDARY)));

  // Align memory to 16-byte boundary for better cache performance
  // The heap-allocated version (using aligned_alloc) works because:
  // The heap has much more available space
  // aligned_alloc properly handles the alignment requirements
  // 16-byte alignment becomes noticeable because it matches SIMD register sizes (like SSE's 128-bit registers)
  float *a = (float *)aligned_alloc(BYTE_BOUNDARY, N * N * sizeof(float));
  float *b = (float *)aligned_alloc(BYTE_BOUNDARY, N * N * sizeof(float));
  float *c = (float *)aligned_alloc(BYTE_BOUNDARY, N * N * sizeof(float));
  float *cc = (float *)aligned_alloc(BYTE_BOUNDARY, N * N * sizeof(float));

  // Initialize matrices with random values
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i * N + j] = (float)rand() / RAND_MAX;
      b[i * N + j] = (float)rand() / RAND_MAX;
      // a[i * N + j] = i * N + j;
      // b[i * N + j] = i * N + j;
      // c[i * N + j] = 0;
      cc[i * N + j] = 0;
    }
  }

  
  for (int i =0; i<2; i++) fast_matmul(a, b, c);
  // for (int i =0; i<10; i++) matmul(a, b, c);

  matmul(a, b, cc);
  for (int i=0; i<(N*N); i++) {
    if (fabsf(cc[i] - c[i]) > 1e-4) {
      printf("MISMATCH at [%d]: %f != %f\n", i, cc[i], c[i]);
      return 0;
    }
  }

  free(a);
  free(b);
  free(c);

  return 0;
}
