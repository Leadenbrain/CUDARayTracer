#ifndef INCLUDE_TEXTURES_SOLID_HPP_
#define INCLUDE_TEXTURES_SOLID_HPP_

#include "texture.hpp"
#include "vec3.hpp"

class solid : public uvTex {
 public:
  __device__ solid() {}
  explicit __device__ solid(const color& c) : col_(c) {}

  __device__ solid(const float& r, const float& g, const float& b)
      : solid(color(r, g, b)) {}

  __device__ virtual color val(const float&,
                               const float&,
                               const vec3&) const override {
    return col_;
  }

 private:
  color col_;
};

#endif  // INCLUDE_TEXTURES_SOLID_HPP_
