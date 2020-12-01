#ifndef INCLUDE_OBJECTS_ISO_FOG_HPP_
#define INCLUDE_OBJECTS_ISO_FOG_HPP_

#include "hit.hpp"
#include "materials/isotropic.hpp"
#include "textures/texture.hpp"
#include "utilities.hpp"
#include "vec3.hpp"

class iso_fog : public hit {
 public:
  __device__ iso_fog(hit* bound,
                     const float& rho,
                     color c,
                     curandState* rand_state)
      : bound_(bound),
        rho_(-1 / rho),
        phase_(new isotropic(c)),
        local_rand_state(rand_state) {}
  __device__ iso_fog(hit* bound,
                     const float& rho,
                     uvTex* c,
                     curandState* rand_state)
      : bound_(bound),
        rho_(-1 / rho),
        phase_(new isotropic(c)),
        local_rand_state(rand_state) {}

  __device__ bool is_hit(const ray&, float, float, hit_rec&) const override;
  __device__ bool bound_box(float t0, float t1, BB& out) const override {
    return bound_->bound_box(t0, t1, out);
  }

 private:
  hit* bound_;
  float rho_;
  material* phase_;
  curandState* local_rand_state;
};

__device__ bool iso_fog::is_hit(const ray& r,
                                float t_min,
                                float t_max,
                                hit_rec& rec) const {
  hit_rec rec0, rec1;
  if (!bound_->is_hit(r, -FLT_MAX, FLT_MAX, rec0))
    return false;

  if (!bound_->is_hit(r, rec0.t + 0.0001f, FLT_MAX, rec1))
    return false;

  if (rec0.t < t_min)
    rec0.t = t_min;
  if (rec1.t > t_max)
    rec1.t = t_max;

  if (rec0.t >= rec1.t)
    return false;

  if (rec0.t < 0)
    rec0.t = 0;

  float r_len = r.direction().norm();
  float d_bound = (rec1.t - rec0.t) * r_len;
  float dist = rho_ * std::log((curand_uniform(local_rand_state)));

  if (dist > d_bound)
    return false;

  rec.t = rec0.t + dist / r_len;
  rec.p = r.at(rec.t);
  rec.n = vec3(1, 0, 0);
  rec.front = true;
  rec.mat = phase_;

  return true;
}

#endif  // INCLUDE_OBJECTS_ISO_FOG_HPP_