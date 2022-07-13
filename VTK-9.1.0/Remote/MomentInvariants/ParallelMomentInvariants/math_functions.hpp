#ifdef __CUDACC__
  // We need this to patch eigen to fix an issue where
  // they require CUDA headers that only exist on < CUDA 9.1
  // to include: https://bitbucket.org/eigen/eigen/commits/034b6c3e101792a3cc3ccabd9bfaddcabe85bb58?at=default
  #include <cuda_runtime.h>
#endif
