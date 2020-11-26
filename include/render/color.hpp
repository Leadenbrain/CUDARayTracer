#ifndef INCLUDE_RENDER_COLOR_HPP_
#define INCLUDE_RENDER_COLOR_HPP_

#include <float.h>
#include <iostream>
#include "utilities.hpp"

#include "materials/diffuse.hpp"
#include "materials/glass.hpp"
#include "materials/metal.hpp"
#include "objects/hit.hpp"
#include "objects/hit_list.hpp"
#include "objects/sphere.hpp"
#include "render/camera.hpp"
#include "render/ray.hpp"
#include "vec3.hpp"

// __device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
//   vec3 oc = r.origin() - center;
//   float a = dot(r.direction(), r.direction());
//   float b = 2.0 * dot(oc, r.direction());
//   float c = dot(oc, oc) - radius * radius;
//   float discriminant = b * b - 4 * a * c;
//   return (discriminant > 0);
// }

__device__ vec3 ray_color(const ray& r,
                          color bg,
                          hit** world,
                          int max_depth,
                          curandState* local_rand_state) {
  // ray temp_r = r;
  // vec3 temp_c(1.0f, 1.0f, 1.0f);

  // for (int i = 0; i < max_depth; i++) {
  //   hit_rec rec;
  //   if (!(*world)->is_hit(temp_r, 0.001f, FLT_MAX, rec)) {
  //     return bg;
  //   } else {
  //     color c;
  //     ray scat;
  //     color light = rec.mat->emit(rec.u, rec.v, rec.p, local_rand_state);

  //     if (!rec.mat->scatter(temp_r, rec, c, scat, local_rand_state)) {
  //       return light;
  //     } else {
  //       return c * temp_c;
  //     }
  //   }
  // }
  // return vec3(0.0f, 0.0f, 0.0f);

  ray temp_r = r;
  vec3 temp_c(1.0f, 1.0f, 1.0f);
  color temp_l;

  for (int i = 0; i < max_depth; i++) {
    hit_rec rec;
    if ((*world)->is_hit(temp_r, 0.001f, FLT_MAX, rec)) {
      ray scat;
      color att;
      if (rec.mat->scatter(temp_r, rec, att, scat, local_rand_state)) {
        temp_c *= att;
        temp_r = scat;
        temp_l = rec.mat->emit(rec.u, rec.v, rec.p, local_rand_state);
        // return temp_c;
      } else {
        temp_l = rec.mat->emit(rec.u, rec.v, rec.p, local_rand_state);
        // temp_c = light;
        return temp_l * temp_c;
      }
    } else {
      // color light = rec.mat->emit(rec.u, rec.v, rec.p, local_rand_state);
      return bg;
    }
  }
  return vec3(0.0f, 0.0f, 0.0f) + temp_l * temp_c;
}
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= max_x) || (j >= max_y))
    return;

  int pxIdx = j * max_x + i;
  // Each thread gets same seed, diff seq number, no offset
  curand_init(1984 + pxIdx, 0, 0, &rand_state[pxIdx]);
}

__global__ void render(vec3* fb,
                       int max_x,
                       int max_y,
                       int ns,
                       int max_depth,
                       camera** cam,
                       hit** world,
                       curandState* rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= max_x) || (j >= max_y)) {
    return;
  }

  int pxIdx = j * max_x + i;
  curandState local_rand_state = rand_state[pxIdx];
  color c(0, 0, 0);
  color bg(0, 0, 0);
  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    // c += ray_color(r, world, max_depth, &local_rand_state);
    c += ray_color(r, bg, world, max_depth, &local_rand_state);
  }
  rand_state[pxIdx] = local_rand_state;
  c /= float(ns);
  c[0] = std::sqrt(c[0]);
  c[1] = std::sqrt(c[1]);
  c[2] = std::sqrt(c[2]);
  fb[pxIdx] = c;
}

__global__ void create_simple_world(hit** d_list,
                                    hit** d_world,
                                    camera** d_camera,
                                    int nx,
                                    int ny) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_list[0] =
        new sphere(vec3(0, 0, -1), 0.5, new diffuse(vec3(0.8, 0.3, 0.3)));
    d_list[1] =
        new sphere(vec3(0, -100.5, -1), 100, new diffuse(vec3(0.8, 0.8, 0.0)));
    d_list[2] =
        new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 1.0));
    d_list[3] = new sphere(vec3(-1, 0, -1), 0.5, new glass(1.5));
    d_list[4] = new sphere(vec3(-1, 0, -1), -0.45, new glass(1.5));
    vec3 from(3, 3, 2);
    vec3 at(0, 0, -1);
    float dist = (from - at).norm();
    *d_world = new hit_list(d_list, 4);
    *d_camera = new camera(from, at, vec3(0, 1, 0), 20.0f,
                           float(nx) / float(ny), 0.001f, dist);
  }
}

__global__ void free_world(hit** d_list, hit** d_world, camera** d_camera) {
  // for (int i = 0; i < 5; i++) {
  //   ((sphere*)d_list[i])->clear_mat();
  //   delete d_list[i];
  // }
  delete *d_world;
  delete *d_camera;
}

#endif  // INCLUDE_RENDER_COLOR_HPP_
