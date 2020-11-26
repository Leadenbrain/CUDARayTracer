#ifndef INCLUDE_OBJECTS_BOUNDING_BOX_HPP_
#define INCLUDE_OBJECTS_BOUNDING_BOX_HPP_

#include <cmath>
#include <utility>
#include "hit.hpp"
#include "render/ray.hpp"
#include "vec3.hpp"

class BB {
 public:
  __device__ BB() {}
  __device__ BB(const point3& min, const point3& max) : min_(min), max_(max) {}

  __device__ inline point3 min() const { return min_; }
  __device__ inline point3 max() const { return max_; }

  __device__ bool is_hit(const ray& r, float t_min, float t_max) const;

  __device__ int axis() const {
    float x = max_.getX() - min_.getX();
    float y = max_.getY() - min_.getY();
    float z = max_.getZ() - min_.getZ();

    if (x > y && x > z)
      return 0;
    else if (y > z)
      return 1;
    else
      return 2;
  }

  __device__ float area() const {
    float x = max_.getX() - min_.getX();
    float y = max_.getY() - min_.getY();
    float z = max_.getZ() - min_.getZ();
    return 2.0f * (x * x + y * y + z * z);
  }

 private:
  point3 min_;
  point3 max_;
};

#ifdef KENSLER
__device__ bool BB::is_hit(const ray& r, float t_min, float t_max) const {
  float ti = t_min;
  float tf = t_max;
  for (int i = 0; i < 3; i++) {
    float invD = 1.0f / r.direction()[i];
    float t0 = (min()[i] - r.origin()[i]) * invD;
    float t1 = (max()[i] - r.origin()[i]) * invD;

    if (invD < 0.0f) {
      float temp_0 = t0;
      t0 = t1;
      t1 = temp_0;
    }
    // std::swap(t0, t1);
    ti = t0 > ti ? t0 : ti;
    tf = t1 < tf ? t1 : tf;
    if (tf <= ti)
      return false;
  }
  return true;
}
#else
__device__ bool BB::is_hit(const ray& r, float t_min, float t_max) const {
  float ti = t_min;
  float tf = t_max;
  for (int i = 0; i < 3; i++) {
    float t0 = fmin((min_[i] - r.origin()[i]) / r.direction()[i],
                    (max_[i] - r.origin()[i]) / r.direction()[i]);
    float t1 = fmax((min_[i] - r.origin()[i]) / r.direction()[i],
                    (max_[i] - r.origin()[i]) / r.direction()[i]);

    ti = fmax(t0, ti);
    tf = fmin(t1, tf);

    if (tf <= ti)
      return false;
  }
  return true;
}
#endif  // KENSLER

__device__ BB surround_box(BB bb0, BB bb1) {
  vec3 small(fmin(bb0.min().getX(), bb1.min().getX()),
             fmin(bb0.min().getY(), bb1.min().getY()),
             fmin(bb0.min().getZ(), bb1.min().getZ()));
  vec3 big(fmax(bb0.max().getX(), bb1.max().getX()),
           fmax(bb0.max().getY(), bb1.max().getY()),
           fmax(bb0.max().getZ(), bb1.max().getZ()));

  return BB(small, big);
}

#endif  // INCLUDE_OBJECTS_BOUNDING_BOX_HPP_
