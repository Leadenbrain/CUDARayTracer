#ifndef INCLUDE_MATERIALS_DIFFUSE_LIGHT_HPP_
#define INCLUDE_MATERIALS_DIFFUSE_LIGHT_HPP_

#include "material.hpp"
#include "textures/solid.hpp"

class diffuse_light : public material {
 public:
  __device__ diffuse_light() {}
  explicit __device__ diffuse_light(color c) : c_(new solid(c)) {}
  explicit __device__ diffuse_light(uvTex* c) : c_(c) {}

  __device__ bool scatter(const ray&,
                          const hit_rec&,
                          color&,
                          ray&,
                          curandState*) const override {
    return false;
  }

  __device__ color emit(const float& u,
                        const float& v,
                        const point3& p,
                        curandState*) const override {
    return c_->val(u, v, p);
  }

 private:
  uvTex* c_;
};

#endif  // INCLUDE_MATERIALS_DIFFUSE_LIGHT_HPP_
