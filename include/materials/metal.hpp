#ifndef INCLUDE_MATERIALS_METAL_HPP_
#define INCLUDE_MATERIALS_METAL_HPP_

#include "material.hpp"

class metal : public material {
 public:
  __device__ metal(const color& a, const float& f)
      : metal_col(a), fuzz_(f < 1 ? f : 1) {}
  __device__ bool scatter(const ray& r,
                          const hit_rec& rec,
                          color& att,
                          ray& scat,
                          curandState* local_rand_state) const override {
    vec3 ref = reflect(unit_v(r.direction()), rec.n);
    scat =
        ray(rec.p, ref + fuzz_ * rand_unit_sphere(local_rand_state), r.time());
    att = metal_col;
    return (dot(scat.direction(), rec.n) > 0.0f);
  }

 private:
  color metal_col;
  float fuzz_;
};

#endif  // INCLUDE_MATERIALS_METAL_HPP_
