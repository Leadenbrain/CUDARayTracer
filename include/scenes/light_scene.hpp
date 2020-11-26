#ifndef INCLUDE_SCENES_LIGHT_SCENE_HPP_
#define INCLUDE_SCENES_LIGHT_SCENE_HPP_

#include "utilities.hpp"

#include "materials/diffuse_light.hpp"
#include "materials/glass.hpp"
#include "materials/metal.hpp"
#include "objects/bvh.hpp"
#include "objects/moving_sphere.hpp"
#include "objects/rectangle.hpp"
#include "render/color.hpp"
#include "textures/checker.hpp"
#include "textures/solid.hpp"
#include "textures/texture.hpp"

__global__ void create_light_scene(hit** d_list,
                                   hit** d_world,
                                   camera** d_camera,
                                   int nx,
                                   int ny,
                                   curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;
    // Ground texture
    checker* ground_tex =
        new checker(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
    diffuse* ground_mat = new diffuse(ground_tex);
    d_list[0] = new sphere(vec3(0, -1000., -1), 1000, ground_mat);

    // Metal texture
    metal* metal_mat = new metal(color(0.7, 0.1, 0.8), 0.3);
    d_list[1] = new sphere(vec3(0, 1, 2), 1, metal_mat);

    // Glass texture
    glass* glass_mat = new glass(1.5);
    d_list[2] = new sphere(vec3(0, 1, 0), 1, glass_mat);

    // Light1 texture
    diffuse_light* light1 = new diffuse_light(color(2, 4, 8));
    d_list[3] = new sphere(vec3(0, 7, 0), 2, light1);

    // Light2 texture
    diffuse_light* light2 = new diffuse_light(color(8, 4, 2));
    d_list[4] = new xy_rectangle(-3, 3, 1, 3, -6, light2);

    vec3 from(10.0f, 2.0f, 3.0f);
    vec3 at(0, 1.0f, 1.0f);
    float dist = (from - at).norm();
    *rand_state = local_rand_state;
    *d_world = new hit_list(d_list, 5);
    *d_camera = new camera(from, at, vec3(0, 1, 0), 20.0f,
                           float(nx) / float(ny), 0.01f, 10.0f);
  }
}

__global__ void free_light_scene(hit** d_list,
                                 hit** d_world,
                                 camera** d_camera) {
  for (int i = 0; i < 4; i++) {
    ((sphere*)d_list[i])->clear_mat();
    delete d_list[i];
  }
  //   (xy_rectangle*)d_list[4]->clear_mat();
  delete d_list[4];
  delete *d_world;
  delete *d_camera;
}

#endif  // INCLUDE_SCENES_LIGHT_SCENE_HPP_
