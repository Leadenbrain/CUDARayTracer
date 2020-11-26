#ifndef INCLUDE_OBJECTS_MOVING_SPHERE_HPP_
#define INCLUDE_OBJECTS_MOVING_SPHERE_HPP_

#include "bounding_box.hpp"
#include "hit.hpp"
#include "vec3.hpp"

class moving_sphere : public hit {
 public:
  __device__ moving_sphere() {}
  __device__ moving_sphere(const point3& c0,
                           const point3& c1,
                           const float& t0,
                           const float& t1,
                           const float& r,
                           material* m)
      : m_(m), c0_(c0), c1_(c1), t0_(t0), t1_(t1), r_(r) {}

  __device__ inline float radius() const { return r_; }
  __device__ inline void clear_mat() { delete m_; }
  __device__ inline point3 center(const float& t) const {
    return c0_ + ((t - t0_) / (t1_ - t0_)) * (c1_ - c0_);
  }

  __device__ bool is_hit(const ray&, float, float, hit_rec& rec) const override;
  __device__ bool bound_box(float, float, BB& out) const override;

 private:
  material* m_;
  point3 c0_, c1_;
  float t0_, t1_, r_;

  __device__ static void get_sphere_uv(const point3& p, float& u, float& v) {
    float t = std::acos(-p.getY());
    float phi = std::atan2(-p.getZ(), p.getX()) * (float)M_PI;

    u = phi / (2.0f * (float)M_PI);
    v = t / (float)M_PI;
  }
};

__device__ bool moving_sphere::is_hit(const ray& r,
                                      float t_min,
                                      float t_max,
                                      hit_rec& rec) const {
  vec3 oc = r.origin() - center(r.time());

  float a = r.direction().norm_sqr();
  float b = dot(oc, r.direction());
  float c = oc.norm_sqr() - r_ * r_;

  float d = b * b - a * c;
  float sqrtd(std::sqrt(d));

#ifdef HEMI_SCAT
  if (d < 0)
    return false;

  float soln = (-b - sqrtd) / a;
  if (soln < t_min || t_max < soln) {
    soln = (-b + sqrtd) / a;
    if (soln < t_min || t_max < soln)
      return false;
  }

  rec.t = soln;
  rec.p = r.at(rec.t);
  vec3 n_out = (rec.p - center(r.time())) / r_;
  rec.set_face(r, n_out);
  rec.mat = m_;
  get_sphere_uv(n_out, rec.u, rec.v);
  return true;
#else
  if (d > 0) {
    float temp = (-b - sqrtd) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
#ifdef MY_GLASS
      vec3 n_out = (rec.p - center(r.time())) / r_;
      rec.set_face(r, n_out);
      get_sphere_uv(n_out, rec.u, rec.v);
#else
      rec.n = (rec.p - center(r.time())) / r_;
#endif  // MY_GLASS
      rec.mat = m_;
      return true;
    }
    temp = (-b + sqrtd) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
#ifdef MY_GLASS
      vec3 n_out = (rec.p - center(r.time())) / r_;
      rec.set_face(r, n_out);
      get_sphere_uv(n_out, rec.u, rec.v);
#else
      rec.n = (rec.p - center(r.time())) / r_;
#endif  // MY_GLASS
      rec.mat = m_;
      return true;
    }
  }
  return false;
#endif  // HEMI_SCAT
}

__device__ bool moving_sphere::bound_box(float t0, float t1, BB& out) const {
  BB b0(center(t0) - vec3(r_, r_, r_), center(t0) + vec3(r_, r_, r_));
  BB b1(center(t1) - vec3(r_, r_, r_), center(t1) + vec3(r_, r_, r_));
  out = surround_box(b0, b1);
  return true;
}

#endif  // INCLUDE_OBJECTS_MOVING_SPHERE_HPP_
