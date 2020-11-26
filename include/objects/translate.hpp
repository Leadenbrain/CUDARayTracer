#ifndef INCLUDE_OBJECTS_TRANSLATE_HPP_
#define INCLUDE_OBJECTS_TRANSLATE_HPP_

#include "hit.hpp"

class translate : public hit {
 public:
  __device__ translate() {}
  __device__ translate(hit* p, const vec3& disp) : p_(p), disp_(disp) {}

  __device__ bool is_hit(const ray& r,
                         float t_min,
                         float t_max,
                         hit_rec& rec) const override {
    ray trans_r(r.origin() - disp_, r.direction(), r.time());
    if (!p_->is_hit(trans_r, t_min, t_max, rec))
      return false;

    rec.p += disp_;
#ifdef MY_GLASS
    rec.set_face(trans_r, rec.n);
#else
    bool front = dot(r.direction(), rec.n) < 0;
    rec.n = front ? rec.n : -rec.n;
#endif  // MY_GLASS
    return true;
  }

  __device__ bool bound_box(float t0, float t1, BB& out) const override {
    if (!p_->bound_box(t0, t1, out))
      return false;

    out = BB(out.min() + disp_, out.max() + disp_);
    return true;
  }

 private:
  hit* p_;
  vec3 disp_;
};

#endif  // INCLUDE_OBJECTS_TRANSLATE_HPP_
