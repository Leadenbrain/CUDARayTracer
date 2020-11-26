#ifndef INCLUDE_MATERIALS_DIFFUSE_HPP_
#define INCLUDE_MATERIALS_DIFFUSE_HPP_

#include "material.hpp"
#include "textures/solid.hpp"
#include "textures/texture.hpp"

class diffuse : public material {
 public:
  __device__ diffuse(color c) : diff_col(new solid(c)) {}
  __device__ diffuse(uvTex* c) : diff_col(c) {}

  __device__ bool scatter(const ray& r,
                          const hit_rec& rec,
                          color& att,
                          ray& scat,
                          curandState* local_rand_state) const override {
    vec3 scat_dir = rec.n + rand_unit_sphere(local_rand_state);
    if (scat_dir.near_null())
      scat_dir = rec.n;

    scat = ray(rec.p, scat_dir, r.time());

    att = diff_col->val(rec.u, rec.v, rec.p);
    return true;
  }

 private:
  uvTex* diff_col;
};

#endif  // INCLUDE_MATERIALS_DIFFUSE_HPP_
