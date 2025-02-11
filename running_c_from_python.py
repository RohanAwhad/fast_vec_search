import os
import ctypes
import subprocess
import sys
import numpy as np

c_code: str = """
#include <stdio.h>

void matmul(float *a, float *b, float *c, int N) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<N; k++) {
        c[i*N+j] += a[i*N+k] * b[k*N+j];
      }
    }
  }
}
"""

c_filename: str = "temp_code.c"
with open(c_filename, "w") as f:
  f.write(c_code)

so_filename: str = "temp_code.so"
try:
  subprocess.check_call(["clang", "-shared", "-o", so_filename, c_filename, "-lm"])
except subprocess.CalledProcessError:
  sys.exit("Compilation of the C code failed.")

lib = ctypes.CDLL(os.path.abspath(so_filename))

lib.matmul.argtypes = [
  np.ctypeslib.ndpointer(dtype=np.float32),
  np.ctypeslib.ndpointer(dtype=np.float32),
  np.ctypeslib.ndpointer(dtype=np.float32),
  ctypes.c_int
]
lib.matmul.restype = None

N = 256
a = np.random.rand(N, N).astype(np.float32)
b = np.random.rand(N, N).astype(np.float32)
result = np.zeros((N, N)).astype(np.float32)

lib.matmul(a, b, result, N)
numpy_result: float = np.matmul(a, b)

print(f"Difference between C and NumPy implementations: {np.sum(abs(result - numpy_result))}")

os.remove(c_filename)
os.remove(so_filename)
