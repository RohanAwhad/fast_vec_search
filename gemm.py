import numpy as np
import time

M = 4096
N = 4096
P = 4096

def np_run():
  
  A = np.random.rand(M, N)
  B = np.random.rand(N, P)
  
  start_time = time.perf_counter()
  C = np.matmul(A, B)
  end_time = time.perf_counter()
  execution_time = end_time - start_time
  C += C
  
  flop = M*P*2*N
  flop_s = flop / execution_time
  print(f'{flop_s/1e9:.2f} GFLOP/s')
  
if __name__ == "__main__":
  print('\n\n===\nNumpy\n===\n')
  for _ in range(10):
    np_run()


# write same for pytorch
import torch
import time

def torch_run():
  
  A = torch.rand(M, N)
  B = torch.rand(N, P)
  
  start_time = time.perf_counter()
  C = torch.matmul(A, B)
  end_time = time.perf_counter()
  execution_time = end_time - start_time
  C += C
  
  flop = M*P*2*N
  flop_s = flop / execution_time
  print(f'{flop_s/1e9:.2f} GFLOP/s')
  
if __name__ == "__main__":
  print('\n\n===\nPyTorch\n===\n')
  for _ in range(10):
    torch_run()
