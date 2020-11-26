#ifndef INCLUDE_TEXTURES_CHECKER_HPP_
#define INCLUDE_TEXTURES_CHECKER_HPP_

#include "solid.hpp"
#include "texture.hpp"

class checker : public uvTex {
 public:
  __device__ checker() {}
  __device__ checker(uvTex* even, uvTex* odd) : even_(even), odd_(odd) {}
  __device__ checker(color c1, color c2)
      : even_(new solid(c1)), odd_(new solid(c2)) {}

  __device__ color val(const float& u,
                       const float& v,
                       const point3& p) const override {
    float s = std::sin(10 * p.getX()) * std::sin(10 * p.getY()) *
              std::sin(10 * p.getZ());
    if (s < 0)
      return odd_->val(u, v, p);
    else
      return even_->val(u, v, p);
  }

 private:
  uvTex* even_;
  uvTex* odd_;
};

#endif  // INCLUDE_TEXTURES_CHECKER_HPP_
