#ifndef INCLUDED_RAY_HPP_
#define INCLUDED_RAY_HPP_

#include "vec3.hpp"

class ray {
 public:
  __device__ ray() {}
  __device__ ray(const vec3& o, const vec3& dir)
      : o_(o), dir_(dir), time_(0.0f) {}
  __device__ ray(const vec3& o, const vec3& dir, const float& time)
      : o_(o), dir_(dir), time_(time) {}

  __device__ point3 origin() const { return o_; }
  __device__ vec3 direction() const { return dir_; }
  __device__ float time() const { return time_; }

  __device__ vec3 at(const float& t) const { return o_ + t * dir_; }

 private:
  point3 o_;
  vec3 dir_;
  float time_;
};

#endif  // INCLUDED_RAY_HPP_
