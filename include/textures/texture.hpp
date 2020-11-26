#ifndef INCLUDE_TEXTURES_TEXTURE_HPP_
#define INCLUDE_TEXTURES_TEXTURE_HPP_

#include "utilities.hpp"
#include "vec3.hpp"

class uvTex {
 public:
  __device__ virtual color val(const float& u,
                               const float& v,
                               const point3& p) const = 0;
};

#endif  // INCLUDE_TEXTURES_TEXTURE_HPP_
