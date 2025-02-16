// gcc -O3 gemm.c
// gcc gemm.c -O2 -ffast-math && ./a.out
// CPU Freq: 3.2GHz

/*
 *
 * perf stat -e cache-misses ./your_program


```assembly
    ldp q0, q1, [x12]          ; Load two quadwords from x12 into registers q0 and q1
    ldp q2, q3, [x12, #16]      ; Load two more quadwords from x12 + 16 into registers q2 and q3

    fmla.4s v4.4s, v5.4s, q0     ; Multiply-accumulate q0 with v5
    fmla.4s v6.4s, v7.4s, q1     ; Multiply-accumulate q1 with v7
    fmla.4s v8.4s, v9.4s, q2     ; Multiply-accumulate q2 with v9
    fmla.4s v10.4s, v11.4s, q3  ; Multiply-accumulate q3 with v11

    stp q4, q5, [x13]          ; Store two quadwords from registers q4 and q5 into x13
    stp q6, q7, [x13, #16]      ; Store two more quadwords from registers q6 and q7 into x13 + 16

    ...
```
 *
 *
 *
* Objdump analysis
*
*
  100003ab4: 3dc01fe3    	ldr	q3, [sp, #112]
  100003ab8: 4f801061    	fmla.4s	v1, v3, v0[0]
  100003abc: 3dc01be3    	ldr	q3, [sp, #96]
  100003ac0: 4fa01061    	fmla.4s	v1, v3, v0[1]
  100003ac4: 3dc017e3    	ldr	q3, [sp, #80]
  100003ac8: 4f801861    	fmla.4s	v1, v3, v0[2]
  100003acc: 3dc013e3    	ldr	q3, [sp, #64]
  100003ad0: 4fa01861    	fmla.4s	v1, v3, v0[3]



16 FLOP/cycle (2x 4 single wide / 4*32 byte FMA)  
Max Clock Frequency: 3200 MHz = 3.2 GHz
Theoretical FLOP/sec = 51.2 GFLOP/s

*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <arm_neon.h>
#include <stdbool.h>

// #define M 256
// #define N 256
// #define P 256

#define M 4096
#define N 4096
#define P 4096

// 16 gives me good speed, but it should be 64 by all the online articles
// #define BYTE_BOUNDARY 16
#define BYTE_BOUNDARY 64
#define BLOCK_I 1024
#define BLOCK_J 16
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
  printf("Validated Matmul : %f GFLOP/s\n", flop_s/(1e9));
}

void preswizzle_b(float *b, float32x4_t *b_swizzled, int bj){
  for (int k=0; k<N; k+=N_LANES) {
    for (int j=0; j<BLOCK_J; j+=WORD_SIZE) {
      for (int l=0; l<N_LANES; l++) {
        int row = (k/N_LANES)*(BLOCK_J*N_LANES/WORD_SIZE);
        int col = (j/WORD_SIZE) * N_LANES + l;
        b_swizzled[row+col] = vld1q_f32(&b[(k+l)*N + (bj+j)]);
      }
    }
  }
}


void fast_matmul(float *a, float *b, float*c) {
  memset(c, 0, M*P*sizeof(float));
  float32x4_t* a_swizzled = (float32x4_t*)aligned_alloc(BYTE_BOUNDARY, (N/N_LANES) * BLOCK_I * sizeof(float32x4_t));
  float32x4_t* b_swizzled = (float32x4_t*)aligned_alloc(BYTE_BOUNDARY, (N/N_LANES) * (BLOCK_J*N_LANES/WORD_SIZE) * sizeof(float32x4_t));
  float32x4_t* tmp = (float32x4_t*)aligned_alloc(BYTE_BOUNDARY, (BLOCK_I * BLOCK_J/4) * sizeof(float32x4_t));

  for (int bi=0; bi<M; bi+=BLOCK_I) {

    // preswizzle a (for contiguous mem array)
    for (int k=0; k<N; k+=N_LANES) {
      for (int i=0; i<BLOCK_I; i++) {
        a_swizzled[(k/N_LANES)*BLOCK_I + i] = vld1q_f32(&a[(bi+i)*N + k]);
      }
    }



    for (int bj=0; bj<P; bj+=BLOCK_J) {
      preswizzle_b(b, b_swizzled, bj);
      
      // Zero tmp using vector operations
      for (int i = 0; i < BLOCK_I * BLOCK_J/4; i++) {
        tmp[i] = vdupq_n_f32(0);
      }

      for (int k=0; k<N; k+=N_LANES) {
        for (int i=0; i<BLOCK_I; i++) {
          // float32x4_t a_vec = vld1q_f32(&a[(bi+i)*N + k]);
          float32x4_t a_vec = a_swizzled[(k/N_LANES)*BLOCK_I + i];
          int row = (k/N_LANES)*(BLOCK_J*N_LANES/WORD_SIZE);
          for (int j=0; j<BLOCK_J; j+=WORD_SIZE) {
            int idx = i*(BLOCK_J/4) + j/4;
            int col = (j/WORD_SIZE) * N_LANES;
            tmp[idx] = vfmaq_laneq_f32(tmp[idx], b_swizzled[row+col+0], a_vec, 0);
            tmp[idx] = vfmaq_laneq_f32(tmp[idx], b_swizzled[row+col+1], a_vec, 1);
            tmp[idx] = vfmaq_laneq_f32(tmp[idx], b_swizzled[row+col+2], a_vec, 2);
            tmp[idx] = vfmaq_laneq_f32(tmp[idx], b_swizzled[row+col+3], a_vec, 3);
          }
        }
      }

      // Store using vector operations
      for (int i=0; i<BLOCK_I; i++) {
        for (int j=0; j<BLOCK_J; j+=4) {
          vst1q_f32(&c[(bi+i)*P+bj+j], tmp[i*(BLOCK_J/4) + j/4]);
        }
      }
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
    printf("     Fast Matmul : %f GFLOP/s\n", flop_s/(1e9));
  }


  matmul(a, b, cc);
  bool is_correct = true;
  for (int i=0; i<(M*P); i++) {
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
