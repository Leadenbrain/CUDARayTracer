#ifndef INCLUDE_SCENES_RANDOM_SCENE_HPP_
#define INCLUDE_SCENES_RANDOM_SCENE_HPP_

#include "utilities.hpp"

#include "objects/bvh.hpp"
#include "objects/moving_sphere.hpp"
#include "textures/checker.hpp"
#include "textures/solid.hpp"
#include "textures/texture.hpp"

__global__ void create_random_world(hit** d_list,
                                    hit** d_world,
                                    camera** d_camera,
                                    int nx,
                                    int ny,
                                    curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;
    // solid* ground_tex = new solid(vec3(0.5, 0.5, 0.5));
    checker* ground_tex =
        new checker(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
    diffuse* ground_mat = new diffuse(ground_tex);
    d_list[0] = new sphere(vec3(0, -1000., -1), 1000.05, ground_mat);

    int i = 1;
    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        float choose_mat = RND;
        vec3 center(a + RND, 0.2, b + RND);
        if (choose_mat < 0.8f) {
          vec3 center2 = center + vec3(0.0f, RND * 0.5f, 0.0f);
          solid* rnd_tex = new solid(vec3(RND * RND, RND * RND, RND * RND));
          diffuse* rnd_mat = new diffuse(rnd_tex);
          d_list[i++] =
              new moving_sphere(center, center2, 0.0, 1.0, 0.2, rnd_mat);
        } else if (choose_mat < 0.95f) {
          d_list[i++] = new sphere(
              center, 0.2,
              new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND),
                             0.5f * (1.0f + RND)),
                        0.5f * RND));
        } else {
          d_list[i++] = new sphere(center, 0.2, new glass(1.5));
        }
      }
    }
    solid* solid_tex = new solid(vec3(0.4, 0.2, 0.1));
    diffuse* sph_mat3 = new diffuse(solid_tex);

    d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new glass(1.5));
    d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, sph_mat3);
    d_list[i++] =
        new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

    vec3 from(13, 2, 3);
    vec3 at(0, 0, 0);
    float dist = (from - at).norm();
    *rand_state = local_rand_state;
    *d_world = new hit_list(d_list, 22 * 22 + 1 + 3);
    // *d_list =
    // new bvh_node(d_list, 0, 22 * 22 + 3, 0.0f, 1.0f, &local_rand_state);
    // hit_list* temp_list = new hit_list(d_list, 22 * 22 + 1 + 3);
    // *d_world = new bvh_node((hit_list*)d_list, 0.0f, 1.0f,
    // &local_rand_state);
    *d_camera = new camera(from, at, vec3(0, 1, 0), 30.0f,
                           float(nx) / float(ny), 0.1f, 10.0f);
  }
}

#endif  // INCLUDE_SCENES_RANDOM_SCENE_HPP_