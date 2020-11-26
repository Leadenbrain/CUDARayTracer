#ifndef INCLUDE_MATERIALS_MATERIAL_HPP_
#define INCLUDE_MATERIALS_MATERIAL_HPP_
#include "utilities.hpp"

#include "objects/hit.hpp"
#include "render/ray.hpp"
#include "vec3.hpp"
// struct hit_rec;

class material {
 public:
  __device__ virtual bool scatter(const ray&,
                                  const hit_rec&,
                                  color&,
                                  ray&,
                                  curandState*) const = 0;
  __device__ virtual color emit(const float&,
                                const float&,
                                const point3&,
                                curandState*) const {
    return color(0.0f, 0.0f, 0.0f);
  }
};

#endif  // INCLUDE_MATERIALS_MATERIAL_HPP_
