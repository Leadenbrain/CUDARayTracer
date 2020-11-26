#ifndef INCLUDE_SCENES_CORNELL_SCENE_HPP_
#define INCLUDE_SCENES_CORNELL_SCENE_HPP_

#include "utilities.hpp"

#include "materials/diffuse_light.hpp"
#include "materials/glass.hpp"
#include "materials/metal.hpp"
#include "objects/bvh.hpp"
#include "objects/cube.hpp"
#include "objects/moving_sphere.hpp"
#include "objects/rectangle.hpp"
#include "objects/translate.hpp"
#include "objects/y_rotation.hpp"
#include "render/color.hpp"
#include "textures/checker.hpp"
#include "textures/solid.hpp"
#include "textures/texture.hpp"

__global__ void create_cornell_scene(hit** d_list,
                                     hit** d_world,
                                     camera** d_camera,
                                     int nx,
                                     int ny,
                                     curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;
    diffuse* red = new diffuse(color(0.65, 0.05, 0.05));
    diffuse* white = new diffuse(color(0.73, 0.73, 0.73));
    diffuse* green = new diffuse(color(0.12, 0.45, 0.15));
    diffuse_light* light = new diffuse_light(color(8, 8, 8));

    d_list[0] = new yz_rectangle(0, 555, 0, 555, 555, red);
    d_list[1] = new yz_rectangle(0, 555, 0, 555, 0, green);
    d_list[2] = new xz_rectangle(213, 343, 227, 332, 554, light);
    d_list[3] = new xz_rectangle(0, 555, 0, 555, 0, white);
    d_list[4] = new xz_rectangle(0, 555, 0, 555, 555, white);
    d_list[5] = new xy_rectangle(0, 555, 0, 555, 555, white);
    hit* b1 = new cube(point3(0, 0, 0), point3(165, 330, 165), white);
    b1 = new y_rotation(b1, 15.0f);
    b1 = new translate(b1, vec3(265, 0, 295));
    d_list[6] = b1;
    hit* b2 = new cube(point3(0, 0, 0), point3(165, 165, 165), white);
    b2 = new y_rotation(b2, -18.0f);
    b2 = new translate(b2, vec3(130, 0, 65));
    d_list[7] = b2;

    vec3 from(278.0f, 278.0f, -800.0f);
    vec3 at(278.0f, 278.0f, 0.0f);
    float dist = (from - at).norm();
    *rand_state = local_rand_state;
    *d_world = new hit_list(d_list, 8);
    *d_camera = new camera(from, at, vec3(0, 1, 0), 40.0f,
                           float(nx) / float(ny), 0.001f, 1000.0f);
  }
}

__global__ void free_cornell_scene(hit** d_list,
                                   hit** d_world,
                                   camera** d_camera) {
  for (int i = 0; i < 8; i++) {
    // ((sphere*)d_list[i])->clear_mat();
    delete d_list[i];
  }
  //   (xy_rectangle*)d_list[4]->clear_mat();
  //   delete d_list[];
  delete *d_world;
  delete *d_camera;
}

#endif  // INCLUDE_SCENES_CORNELL_SCENE_HPP_
