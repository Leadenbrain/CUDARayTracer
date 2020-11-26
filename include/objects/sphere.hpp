#ifndef INCLUDE_OBJECTS_SPHERE_HPP_
#define INCLUDE_OBJECTS_SPHERE_HPP_

#include "bounding_box.hpp"
#include "hit.hpp"
#include "render/ray.hpp"
#include "vec3.hpp"

class sphere : public hit {
 public:
  __device__ sphere() {}
  __device__ sphere(const point3& c, const float& r, material* m)
      : m_(m), c_(c), r_(r) {}

  __device__ inline float radius() const { return r_; }
  __device__ inline point3 center() const { return c_; }
  __device__ inline void clear_mat() { delete m_; }

  __device__ bool is_hit(const ray&, float, float, hit_rec&) const override;
  __device__ bool bound_box(float, float, BB& out) const override {
    out = BB(c_ - vec3(r_, r_, r_), c_ + vec3(r_, r_, r_));
    return true;
  };

 private:
  float r_;
  point3 c_;
  material* m_;

  __device__ static void get_sphere_uv(const point3& p, float& u, float& v) {
    float t = std::acos(-p.getY());
    float phi = std::atan2(-p.getZ(), p.getX()) * (float)M_PI;

    u = phi / (2.0f * (float)M_PI);
    v = t / (float)M_PI;
  }
};

__device__ bool sphere::is_hit(const ray& r,
                               float t_min,
                               float t_max,
                               hit_rec& rec) const {
  vec3 oc = r.origin() - c_;

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
  vec3 n_out = (rec.p - c_) / r_;
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
      vec3 n_out = (rec.p - c_) / r_;
      rec.set_face(r, n_out);
      get_sphere_uv(n_out, rec.u, rec.v);
#else
      rec.n = (rec.p - c_) / r_;
#endif  // MY_GLASS
      rec.mat = m_;
      return true;
    }
    temp = (-b + sqrtd) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
#ifdef MY_GLASS
      vec3 n_out = (rec.p - c_) / r_;
      rec.set_face(r, n_out);
      get_sphere_uv(n_out, rec.u, rec.v);
#else
      rec.n = (rec.p - c_) / r_;
#endif  // MY_GLASS
      rec.mat = m_;
      return true;
    }
  }
  return false;
#endif  // HEMI_SCAT
}

#endif  // INCLUDE_OBJECTS_SPHERE_HPP_
