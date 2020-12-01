#include <ctime>
#include <iostream>
#include "utilities.hpp"
#include "vec3.hpp"

#include "render/color.hpp"
#include "render/camera.hpp"
#include "objects/sphere.hpp"
#include "objects/hit_list.hpp"

#include "scenes/random_scene.hpp"
#include "scenes/cornell_scene.hpp"
#include "scenes/light_scene.hpp"


int main(void) {
  float aspect{16.0 / 16.0};
  int width = 600;
  int height = static_cast<int>(width/aspect);
  int ns = 500;
  int max_depth = 50;
  int tx = _TX;
  int ty = _TX;

  size_t fb_size = width * height * sizeof(vec3);

  vec3* fb;
  curandState *d_rand_state;
  curandState *d_rand_state2;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, width*height*sizeof(curandState)));
  checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1*sizeof(curandState)));
  
  rand_init<<<1,1>>>(d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  hit** d_list;
  int n_hit = 8;
  hit** d_world;
  camera** d_camera;
  checkCudaErrors(cudaMalloc((void**)&d_list, n_hit*sizeof(hit*)));
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hit*)));
  checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
  create_cornell_scene<<<1,1>>>(d_list,d_world,d_camera, width, height, d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  std::cerr << "Rendering: (" << width << ", " << height << ") \n";

  std::cerr << "In: (" << tx << ", " << ty << ") \n";

  clock_t start, stop;
  start = clock();

  // TODO:
  // Can convert float to 8-bit component when sending back for improved
  // performance
  dim3 blocks(width / tx + 1, height / ty + 1);
  dim3 threads(tx, ty);
  render_init<<<blocks, threads>>>(width, height, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  render<<<blocks, threads>>>(fb, width, height, ns, max_depth, d_camera, d_world, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stop = clock();
  double t_sec = (static_cast<double>(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "Finished Render in : " << t_sec << "\n";

  std::cout << "P3\n" << width << " " << height << "\n255\n";

  for (int j = height - 1; j >= 0; j--) {
    for (int i = 0; i < width; i++) {
      size_t pixel_index = j * width + i;
      float r = fb[pixel_index].getX();
      float g = fb[pixel_index].getY();
      float b = fb[pixel_index].getZ();
      r = my_clamp(r,0.000f,0.999f);
      g = my_clamp(g,0.000f,0.999f);
      b = my_clamp(b,0.000f,0.999f);
      int ir = static_cast<int>(256 * r);
      int ig = static_cast<int>(256 * g);
      int ib = static_cast<int>(256 * b);
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }

  // checkCudaErrors(cudaDeviceSynchronize());
  free_cornell_scene<<<1,1>>>(d_list,d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(fb));
}