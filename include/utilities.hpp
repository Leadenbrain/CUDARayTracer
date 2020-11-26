#ifndef INCLUDE_UTILITIES_HPP_
#define INCLUDE_UTILITIES_HPP_

#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define checkCudaErrors(x) check_cuda((x), #x, __FILE__, __LINE__)

#define RANDVEC3                                                           \
  vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), \
       curand_uniform(local_rand_state))

#define RND (curand_uniform(&local_rand_state))

void check_cuda(cudaError_t result,
                const char* func,
                const char* file,
                const int line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << " : " << line << " '" << func << "' \n";
    cudaDeviceReset();
    exit(99);
  }
}

__global__ void rand_init(curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, rand_state);
  }
}

__host__ __device__ inline float clamp(float x, float min, float max) {
  if (x < min)
    return min;
  if (x > max)
    return max;
  return x;
}

__device__ inline float deg_to_rad(const float& t) {
  return t * (float)M_PI / 180.0f;
}

#endif  // INCLUDE_UTILITIES_HPP_
