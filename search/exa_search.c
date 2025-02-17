// gcc exa_search.c -framework Accelerate && ./a.out

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>

#define EMBED_SIZE 256
#define BYTE_SIZE 8
#define SUBVECTOR_SIZE 8

void get_binary_matrix(float *binary_array) {
  // 256 because 2 ** 8 combinations
  for (int num=0; num < 256; num++) {
    for (int i = BYTE_SIZE - 1; i >= 0; i--) {
      int bit = (num >> i) & 1;
      binary_array[num*BYTE_SIZE + BYTE_SIZE - 1 - i] = (bit == 0) ? -1.0 : 1.0;
    }
  }
}

void compile_scores(uint8_t *compressed, float *lookup_table, float *scores, int DB_SIZE) {
  for (int i=0; i<DB_SIZE; i++) {
    float sc = 0.0;
    for (int j=0; j<EMBED_SIZE/BYTE_SIZE; j++) {
      int key = (int)compressed[i*EMBED_SIZE/BYTE_SIZE + j];
      sc += lookup_table[j*256+key];
    }
    scores[i] = sc;
  }
}

void quantize(uint8_t *compressed, float *db, int DB_SIZE) {
  for (int i=0; i<DB_SIZE; i++) {
    for (int j=0; j<EMBED_SIZE/BYTE_SIZE; j++) {
      unsigned char byte = 0;
      for (int k=0; k<BYTE_SIZE; k++) {
        if (db[i*EMBED_SIZE + j*BYTE_SIZE + k] > 0) {
          byte |= (1 << (BYTE_SIZE - k + 1));
        }
      }
      compressed[i*EMBED_SIZE/BYTE_SIZE + j] = (uint8_t) byte;
    }
  }
}

void get_scores(float *query, uint8_t *compressed, float *matrix_B, float *scores, int n, int DB_SIZE) {
  // n is number of query embeddings
  float lookup_table[n*EMBED_SIZE/SUBVECTOR_SIZE][256];
  vDSP_mmul((float *)query, 1, (float *) matrix_B, 1, (float *)lookup_table, 1, n*EMBED_SIZE/SUBVECTOR_SIZE, 256, BYTE_SIZE);
  compile_scores((uint8_t *)compressed, (float *)lookup_table, (float *)scores, DB_SIZE);
}

// ===
// Main
// ===

int main() {
  int DB_SIZE = 100;

  float temp[EMBED_SIZE];
  float db[DB_SIZE * EMBED_SIZE];
  uint8_t compressed[DB_SIZE * EMBED_SIZE / BYTE_SIZE];

  // random filling
  for (int i=0; i<DB_SIZE*EMBED_SIZE; i++) { db[i] = ((float) rand() / RAND_MAX) * 2 -1; }
  // quantize
  quantize(compressed, db, DB_SIZE);


  // generate subvector scores
  float matrix_B[256][8];
  get_binary_matrix((float *) matrix_B);
  float query[EMBED_SIZE];
  for (int i=0; i<EMBED_SIZE; i++) { query[i] = ((float) rand() / RAND_MAX) * 2 - 1; }
  // get subvectors lookup_table
  float scores[DB_SIZE];
  get_scores(query, compressed, (float *)matrix_B, scores, 1, DB_SIZE);
  // compile_scores

  printf("binary dot product scores:\n");
  for (int i=0; i<DB_SIZE; i++) { printf("%5.2f ", scores[i]); }
  printf("\n");

  float exact_scores[DB_SIZE];
  vDSP_mmul((float *)query, 1, db, 1, (float *)exact_scores, 1, 1, DB_SIZE, EMBED_SIZE);
  printf("float dot product scores:\n");
  for (int i=0; i<DB_SIZE; i++) { printf("%5.2f ", exact_scores[i]); }
  printf("\n");


  // argsort and print top 10 ids in descending score order. have separate for binary and float
  float binary_sorted[DB_SIZE];
  int binary_indices[DB_SIZE];
  float float_sorted[DB_SIZE];
  int float_indices[DB_SIZE];

  for (int i = 0; i < DB_SIZE; i++) {
    binary_sorted[i] = scores[i];
    float_sorted[i] = exact_scores[i];
    binary_indices[i] = i;
    float_indices[i] = i;
  }

  for (int i = 0; i < DB_SIZE - 1; i++) {
    for (int j = 0; j < DB_SIZE - i - 1; j++) {
      if (binary_sorted[j] < binary_sorted[j + 1]) {
        float temp = binary_sorted[j];
        binary_sorted[j] = binary_sorted[j + 1];
        binary_sorted[j + 1] = temp;
        int temp_idx = binary_indices[j];
        binary_indices[j] = binary_indices[j + 1];
        binary_indices[j + 1] = temp_idx;
      }
      if (float_sorted[j] < float_sorted[j + 1]) {
        float temp = float_sorted[j];
        float_sorted[j] = float_sorted[j + 1];
        float_sorted[j + 1] = temp;
        int temp_idx = float_indices[j];
        float_indices[j] = float_indices[j + 1];
        float_indices[j + 1] = temp_idx;
      }
    }
  }

  printf("\nTop binary scores and indices:\n");
  for (int i = 0; i < 10; i++) {
    printf("%2d ", binary_indices[i]);
  }
  printf("\n");

  printf("\nTop float scores and indices:\n");
  for (int i = 0; i < 10; i++) {
    printf("%2d ", float_indices[i]);
  }
  printf("\n");


  return 0;
}

// i want to compile this into a shared object, so that i can use it in python
// Here's how to modify and compile it as a shared library:
//
// 1. Modify the code to expose functions:
// ```c
// #include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <Accelerate/Accelerate.h>
//
// // Add this at the top
// #ifdef __cplusplus
// extern "C" {
// #endif
//
// // Export these functions
// void quantize(uint8_t *compressed, float *db, int DB_SIZE);
// void get_scores(float *query, uint8_t *compressed, float *matrix_B, float *scores, int n, int DB_SIZE);
//
// #ifdef __cplusplus
// }
// #endif
// ```
//
// 2. Remove the main function and save as `exa_search.c`
//
// 3. Compile command:
// ```bash
// gcc -c -fPIC exa_search.c -framework Accelerate
// gcc -shared -o libexa_search.so exa_search.o -framework Accelerate
// ```
//
// 4. Python usage:
// ```python
// import ctypes
// import numpy as np
//
// lib = ctypes.CDLL('./libexa_search.so')
//
// # Define argument types
// lib.quantize.argtypes = [
//   np.ctypeslib.ndpointer(dtype=np.uint8),
//   np.ctypeslib.ndpointer(dtype=np.float32),
//   ctypes.c_int
// ]
//
// lib.get_scores.argtypes = [
//   np.ctypeslib.ndpointer(dtype=np.float32),
//   np.ctypeslib.ndpointer(dtype=np.uint8),
//   np.ctypeslib.ndpointer(dtype=np.float32),
//   np.ctypeslib.ndpointer(dtype=np.float32),
//   ctypes.c_int,
//   ctypes.c_int
// ]
// ```
//
// Move shared library to where Python can find it or specify full path when loading.
