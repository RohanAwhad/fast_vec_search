# write a main function that will create 2 np arrays of size NxN and matmul them. Also time the thing in seconds
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time

def main():
  N = 5120
  
  A = np.random.rand(N, N)
  B = np.random.rand(N, N)
  
  start_time = time.perf_counter()
  C = np.matmul(A, B)
  end_time = time.perf_counter()
  execution_time = end_time - start_time
  C += C
  
  flop = N*N*2*N
  flop_s = flop / execution_time
  print(f'{flop_s/1e9:.2f} GFLOP/s')
  
if __name__ == "__main__":
  for _ in range(1000000):
    main()
