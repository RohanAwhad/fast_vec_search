// gcc exa_search.c -framework Accelerate && ./a.out
// gcc -c -O2 -ffast-math -fPIC exa_search.c && gcc -shared -o libexa_search.so exa_search.o -framework Accelerate

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

void compile_scores(uint8_t *compressed, float *lookup_table, float *scores, int n, int DB_SIZE) {
  for (int k=0; k<n; k++) {  // looping over n queries

    for (int i=0; i<DB_SIZE; i++) {  // for each query loop over entire db

      float sc = 0.0;
      for (int j=0; j<EMBED_SIZE/BYTE_SIZE; j++) {
        int key = (int)compressed[i*EMBED_SIZE/BYTE_SIZE + j];  // lookup key from compressed db

        // given key and query get score from lookup table
        int idx = k*256*EMBED_SIZE/BYTE_SIZE + j*256 + key;
        sc += lookup_table[idx];
      }

      scores[k*DB_SIZE + i] = sc;
    }

  }
}


void quantize(uint8_t *compressed, float *db, int DB_SIZE) {
  for (int i = 0; i < DB_SIZE; i++) {
    for (int j = 0; j < EMBED_SIZE; j += BYTE_SIZE) {
      uint8_t byte = 0;
      for (int k = 0; k < BYTE_SIZE; k++) {
        byte = (byte << 1) | (db[i * EMBED_SIZE + j + k] >= 0);
      }
      compressed[i * (EMBED_SIZE/BYTE_SIZE) + (j/BYTE_SIZE)] = byte;
    }
  }
}

void instantiate_lookup_table(float *lookup_table, float *query, float *matrix_B, int n) {
  vDSP_mmul(
    query,                          // A
    1,                             // stride A
    matrix_B,                      // B
    1,                             // stride B
    lookup_table,                  // C
    1,                             // stride C
    n*EMBED_SIZE/SUBVECTOR_SIZE,     // M (rows in A)
    256,                           // N (cols in B)
    SUBVECTOR_SIZE                 // P (cols in A, rows in B)
  );
}

