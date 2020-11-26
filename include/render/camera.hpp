#ifndef INCLUDE_RENDER_CAMERA_HPP_
#define INCLUDE_RENDER_CAMERA_HPP_

#include "ray.hpp"
#include "utilities.hpp"
#include "vec3.hpp"

class camera {
 public:
  __device__ camera(const point3& from,
                    const point3& to,
                    const vec3& vup,
                    const float& fov,
                    const float& ar,
                    const float& ap,
                    const float& focus,
                    const float& t0 = 0.0f,
                    const float& t1 = 0.0f) {
    float t = deg_to_rad(fov);
    float h = std::tan(t / 2.0f);
    float vp_height = 2.0f * h;
    float vp_width = ar * vp_height;

    w_ = unit_v(from - to);
    u_ = unit_v(cross(vup, w_));
    v_ = cross(w_, u_);

    t0_ = t0;
    t1_ = t1;

    o_ = from;
    hor_ = focus * vp_width * u_;
    ver_ = focus * vp_height * v_;

    llc_ = o_ - hor_ / 2.0f - ver_ / 2.0f - focus * w_;
    r_ = ap / 2.0f;
  }

  __device__ ray get_ray(const float& s,
                         const float& t,
                         curandState* local_rand_state) {
    vec3 rd = r_ * rand_unit_disk(local_rand_state);
    vec3 off = u_ * rd.getX() + v_ * rd.getY();

    float rand = curand_uniform(local_rand_state);
    rand *= (t1_ - t0_ + 0.999999);
    rand += t0_;

    return ray(o_ + off, llc_ + s * hor_ + t * ver_ - o_ - off, rand);
  }

 private:
  point3 o_, llc_;
  vec3 hor_, ver_;
  vec3 u_, v_, w_;
  float r_, t0_, t1_;
};

#endif  // INCLUDE_RENDER_CAMERA_HPP_
