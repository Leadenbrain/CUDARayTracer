#ifndef INCLUDE_OBJECTS_RECTANGLE_HPP_
#define INCLUDE_OBJECTS_RECTANGLE_HPP_

#include "bounding_box.hpp"
#include "hit.hpp"
#include "render/ray.hpp"
#include "vec3.hpp"

class xy_rectangle : public hit {
 public:
  __device__ xy_rectangle() {}
  __device__ xy_rectangle(const float& x0,
                          const float& x1,
                          const float& y0,
                          const float& y1,
                          const float& k,
                          material* m)
      : x0_(x0), x1_(x1), y0_(y0), y1_(y1), k_(k), m_(m) {}

  __device__ inline void clear_mat() { delete m_; }

  __device__ bool is_hit(const ray&, float, float, hit_rec& hit) const override;
  __device__ bool bound_box(float, float, BB& out) const override {
    out = BB(point3(x0_, y0_, k_ - 0.0001), point3(x1_, y1_, k_ + 0.0001));
    return true;
  }

 private:
  float x0_, x1_, y0_, y1_, k_;
  material* m_;
};

__device__ bool xy_rectangle::is_hit(const ray& r,
                                     float t_min,
                                     float t_max,
                                     hit_rec& rec) const {
  float t = (k_ - r.origin().getZ()) / r.direction().getZ();
  if (t < t_min || t > t_max)
    return false;
  float x = r.origin().getX() + t * r.direction().getX();
  float y = r.origin().getY() + t * r.direction().getY();

  if (x < x0_ || x > x1_ || y < y0_ || y > y1_)
    return false;

  rec.u = (x - x0_) / (x1_ - x0_);
  rec.v = (y - y0_) / (y1_ - y0_);
  rec.t = t;
#ifdef MY_GLASS
  vec3 n_out = vec3(0.0f, 0.0f, 1.0f);
  rec.set_face(r, n_out);
#else
  rec.n = vec3(0.0f, 0.0f, 1.0f);
#endif  // MY_GLASS
  rec.mat = m_;
  rec.p = r.at(t);

  return true;
}

class xz_rectangle : public hit {
 public:
  __device__ xz_rectangle() {}
  __device__ xz_rectangle(const float& x0,
                          const float& x1,
                          const float& z0,
                          const float& z1,
                          const float& k,
                          material* m)
      : x0_(x0), x1_(x1), z0_(z0), z1_(z1), k_(k), m_(m) {}

  __device__ inline void clear_mat() { delete m_; }

  __device__ bool is_hit(const ray&, float, float, hit_rec& hit) const override;
  __device__ bool bound_box(float, float, BB& out) const override {
    out = BB(point3(x0_, k_ - 0.0001, z0_), point3(x1_, k_ + 0.0001, z1_));
    return true;
  }

 private:
  float x0_, x1_, z0_, z1_, k_;
  material* m_;
};

__device__ bool xz_rectangle::is_hit(const ray& r,
                                     float t_min,
                                     float t_max,
                                     hit_rec& rec) const {
  float t = (k_ - r.origin().getY()) / r.direction().getY();
  if (t < t_min || t > t_max)
    return false;
  float x = r.origin().getX() + t * r.direction().getX();
  float z = r.origin().getZ() + t * r.direction().getZ();

  if (x < x0_ || x > x1_ || z < z0_ || z > z1_)
    return false;

  rec.u = (x - x0_) / (x1_ - x0_);
  rec.v = (z - z0_) / (z1_ - z0_);
  rec.t = t;
#ifdef MY_GLASS
  vec3 n_out = vec3(0.0f, 1.0f, 0.0f);
  rec.set_face(r, n_out);
#else
  rec.n = vec3(0.0f, 1.0f, 0.0f);
#endif  // MY_GLASS
  rec.mat = m_;
  rec.p = r.at(t);

  return true;
}

class yz_rectangle : public hit {
 public:
  __device__ yz_rectangle() {}
  __device__ yz_rectangle(const float& y0,
                          const float& y1,
                          const float& z0,
                          const float& z1,
                          const float& k,
                          material* m)
      : y0_(y0), y1_(y1), z0_(z0), z1_(z1), k_(k), m_(m) {}

  __device__ inline void clear_mat() { delete m_; }

  __device__ bool is_hit(const ray&, float, float, hit_rec& hit) const override;
  __device__ bool bound_box(float, float, BB& out) const override {
    out = BB(point3(k_ - 0.0001, y0_, z0_), point3(k_ + 0.0001, y1_, z0_));
    return true;
  }

 private:
  float y0_, y1_, z0_, z1_, k_;
  material* m_;
};

__device__ bool yz_rectangle::is_hit(const ray& r,
                                     float t_min,
                                     float t_max,
                                     hit_rec& rec) const {
  float t = (k_ - r.origin().getX()) / r.direction().getX();
  if (t < t_min || t > t_max)
    return false;
  float y = r.origin().getY() + t * r.direction().getY();
  float z = r.origin().getZ() + t * r.direction().getZ();

  if (y < y0_ || y > y1_ || z < z0_ || z > z1_)
    return false;

  rec.u = (y - y0_) / (y1_ - y0_);
  rec.v = (z - z0_) / (z1_ - z0_);
  rec.t = t;
#ifdef MY_GLASS
  vec3 n_out = vec3(1.0f, 0.0f, 0.0f);
  rec.set_face(r, n_out);
#else
  rec.n = vec3(1.0f, 0.0f, 0.0f);
#endif  // MY_GLASS
  rec.mat = m_;
  rec.p = r.at(t);

  return true;
}

#endif  // INCLUDE_OBJECTS_RECTANGLE_HPP_
