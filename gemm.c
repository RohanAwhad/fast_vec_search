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

// #define N 2048
#define N 4096
#define BYTE_BOUNDARY 16

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
  int BLOCK_I = 128;
  int BLOCK_J = 32;
  // int BLOCK_I = 4;
  // int BLOCK_J = 8;
  int N_LANES = 4;
  int WORD_SIZE = 4;
  double start = get_time();
  memset(c, 0, N*N*sizeof(float));


  float32x4_t* b_swizzled = (float32x4_t*)aligned_alloc(BYTE_BOUNDARY, (N/N_LANES) * (BLOCK_J*N_LANES/WORD_SIZE) * sizeof(float32x4_t));

  float *tmp = (float *)aligned_alloc(BYTE_BOUNDARY, BLOCK_I * BLOCK_J * sizeof(float));
  for (int bi=0; bi<N; bi+=BLOCK_I) {
    for (int bj=0; bj<N; bj+=BLOCK_J) {

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

      // for (int i = 0; i < (N/N_LANES); i++) {
      //   for (int j = 0; j < (BLOCK_J*N_LANES/WORD_SIZE); j++) {
      //     float32x4_t v = b_swizzled[i*N + j];
      //     printf("%.2f %.2f %.2f %.2f | ", 
      //       vgetq_lane_f32(v, 0),
      //       vgetq_lane_f32(v, 1),
      //       vgetq_lane_f32(v, 2),
      //       vgetq_lane_f32(v, 3));
      //   }
      //   printf("\n");
      // }
      //
      // // Wait for keypress
      // printf("Press enter to continue...");
      // getchar();
      // // loading is correct (as i imagined!)

      // compute
      memset(tmp, 0, BLOCK_I * BLOCK_J * sizeof(float));
      for (int k=0; k<N; k+=N_LANES) {

        for (int i=0; i<BLOCK_I; i++) {
          float32x4_t a_vec = vld1q_f32(&a[(bi+i)*N+k]);

          int row = (k/N_LANES)*(BLOCK_J*N_LANES/WORD_SIZE);
          for (int j=0; j<BLOCK_J; j+=WORD_SIZE) {

            float32x4_t acc = vld1q_f32(&tmp[i*BLOCK_J+j]);

            // float32x4_t b_vec_0 = vld1q_f32(&b[(k)*N+bj+j]);
            // float32x4_t b_vec_1 = vld1q_f32(&b[(k+1)*N+bj+j]);
            // float32x4_t b_vec_2 = vld1q_f32(&b[(k+2)*N+bj+j]);
            // float32x4_t b_vec_3 = vld1q_f32(&b[(k+3)*N+bj+j]);
            

            // acc = vfmaq_laneq_f32(acc, b_vec_0, a_vec, 0);
            // acc = vfmaq_laneq_f32(acc, b_vec_1, a_vec, 1);
            // acc = vfmaq_laneq_f32(acc, b_vec_2, a_vec, 2);
            // acc = vfmaq_laneq_f32(acc, b_vec_3, a_vec, 3);


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
      for (int i=0; i<BLOCK_I; i++) {
        for (int j=0; j<BLOCK_J; j++) {
          c[(bi+i)*N+bj+j] = tmp[i*BLOCK_J+j];
        }
      }


    }
  }

// Pre-swizzle in 4x4 blocks
// for (int k = 0; k < N; k += 4) {
//   for (int j = 0; j < N; j += 4) {
//     float32x4_t row0 = vld1q_f32(&b[(k+0)*N + j]);
//     float32x4_t row1 = vld1q_f32(&b[(k+1)*N + j]);
//     float32x4_t row2 = vld1q_f32(&b[(k+2)*N + j]);
//     float32x4_t row3 = vld1q_f32(&b[(k+3)*N + j]);
//     
//     // Store transposed
//     vst1q_f32(&b_swizzled[(j*N + k*4)], row0);
//     vst1q_f32(&b_swizzled[(j*N + k*4) + 4], row1);
//     vst1q_f32(&b_swizzled[(j*N + k*4) + 8], row2);
//     vst1q_f32(&b_swizzled[(j*N + k*4) + 12], row3);
//   }
// }

  free(tmp);
  free(b_swizzled);

  double end = get_time();
  float time_spent = (float) (end - start);
  float flop = N*N*(2.0*N);
  float flop_s = flop / time_spent;
  printf("   Fast Matmul : %f GFLOP/s\n", flop_s/(1e9));
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
      cc[i * N + j] = 0;
    }
  }


  
  for (int i =0; i<10; i++) fast_matmul(a, b, c);
  // for (int i =0; i<10; i++) matmul(a, b, c);

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
