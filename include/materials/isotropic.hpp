#ifndef INCLUDE_MATERIALS_ISOTROPIC_HPP_
#define INCLUDE_MATERIALS_ISOTROPIC_HPP_

#include "material.hpp"
#include "textures/solid.hpp"
#include "utilities.hpp"
#include "vec3.hpp"

class isotropic : public material {
 public:
  __device__ explicit isotropic(color c) : c_(new solid(c)) {}
  __device__ explicit isotropic(uvTex* c) : c_(c) {}

  __device__ bool scatter(const ray& r,
                          const hit_rec& rec,
                          color& att,
                          ray& scat,
                          curandState* local_rand_state) const override {
    scat = ray(rec.p, rand_sphere(local_rand_state), r.time());
    att = c_->val(rec.u, rec.v, rec.p);
    return true;
  }

 private:
  uvTex* c_;
};

#endif  // INCLUDE_MATERIALS_ISOTROPIC_HPP_