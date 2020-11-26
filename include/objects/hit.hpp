#ifndef INCLUDE_OBJECTS_HIT_HPP_
#define INCLUDE_OBJECTS_HIT_HPP_

#include "bounding_box.hpp"
#include "render/ray.hpp"
#include "vec3.hpp"

class material;
class BB;

struct hit_rec {
  point3 p;
  vec3 n;
  material* mat;
  float t, u, v;
#ifdef MY_GLASS
  bool front;

  __device__ void set_face(const ray& r, const vec3& n_out) {
    front = dot(r.direction(), n_out) < 0;
    n = front ? n_out : -n_out;
  }
#endif  // MY_GLASS
};

class hit {
 public:
  //  Shouldn't be virtual, gives stack size warning, but also cant not :/
  __device__ virtual bool is_hit(const ray&, float, float, hit_rec&) const = 0;
  __device__ virtual bool bound_box(float, float, BB&) const = 0;
};

#endif  // INCLUDE_OBJECTS_HIT_HPP_
