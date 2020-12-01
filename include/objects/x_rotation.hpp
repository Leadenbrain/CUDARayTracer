#ifndef INCLUDE_OBJECTS_X_ROTATION_HPP_
#define INCLUDE_OBJECTS_X_ROTATION_HPP_

#include "hit.hpp"

class x_rotation : public hit {
 public:
  __device__ x_rotation();
  __device__ x_rotation(hit* p, const float& t_deg);

  __device__ bool is_hit(const ray& r,
                         float t_min,
                         float t_max,
                         hit_rec& rec) const override;
  __device__ bool bound_box(float, float, BB& out) const override {
    out = bound_;
    return box_bool;
  }

 private:
  hit* p_;
  float sin_t, cos_t;
  bool box_bool;
  BB bound_;
};

__device__ x_rotation::x_rotation(hit* p, const float& t_deg) : p_(p) {
  float t_rad = deg_to_rad(t_deg);
  sin_t = std::sin(t_rad);
  cos_t = std::cos(t_rad);
  box_bool = p_->bound_box(0, 1, bound_);

  point3 min(FLT_MIN, FLT_MIN, FLT_MIN);
  point3 max(FLT_MAX, FLT_MAX, FLT_MAX);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        float x = i * bound_.max().getX();
        float y = j * bound_.max().getY();
        float z = k * bound_.max().getZ();

        float y_out = cos_t * y - sin_t * z;
        float z_out = sin_t * y + cos_t * z;

        vec3 out(x, y_out, z_out);

        for (int a = 0; a < 3; a++) {
          min[a] = fmin(min[a], out[a]);
          max[a] = fmax(max[a], out[a]);
        }
      }
    }
  }

  bound_ = BB(min, max);
}

__device__ bool x_rotation::is_hit(const ray& r,
                                   float t_min,
                                   float t_max,
                                   hit_rec& rec) const {
  point3 o = r.origin();
  vec3 d = r.direction();

  // Want to say this is right, might have my visual rotated
  o[1] = cos_t * r.origin()[1] + sin_t * r.origin()[2];
  o[2] = -sin_t * r.origin()[1] + cos_t * r.origin()[2];

  d[1] = cos_t * r.direction()[1] + sin_t * r.direction()[2];
  d[2] = -sin_t * r.direction()[1] + cos_t * r.direction()[2];

  ray rot_r(o, d, r.time());

  if (!p_->is_hit(rot_r, t_min, t_max, rec))
    return false;

  point3 p = rec.p;
  vec3 n = rec.n;

  p[1] = cos_t * rec.p[1] - sin_t * rec.p[2];
  p[2] = sin_t * rec.p[1] + cos_t * rec.p[2];

  n[1] = cos_t * rec.n[1] - sin_t * rec.n[2];
  n[2] = sin_t * rec.n[1] + cos_t * rec.n[2];

  rec.p = p;
#ifdef MY_GLASS
  rec.set_face(rot_r, n);
#else
  bool front = dot(r.direction(), rec.n) < 0;
  rec.n = front ? rec.n : -rec.n;
#endif  // MY_GLASS

  return true;
}

#endif  // INCLUDE_OBJECTS_X_ROTATION_HPP_
