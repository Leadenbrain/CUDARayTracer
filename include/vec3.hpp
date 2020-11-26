#ifndef INCLUDE_VEC3_HPP_
#define INCLUDE_VEC3_HPP_

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "utilities.hpp"

class vec3 {
 public:
  __host__ __device__ vec3() : e{0, 0, 0} {}
  __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}
  __host__ __device__ inline float getX() const { return e[0]; }
  __host__ __device__ inline float getY() const { return e[1]; }
  __host__ __device__ inline float getZ() const { return e[2]; }
  __host__ __device__ inline void setX(std::istream& is) { is >> e[0]; }
  __host__ __device__ inline void setY(std::istream& is) { is >> e[1]; }
  __host__ __device__ inline void setZ(std::istream& is) { is >> e[2]; }

  __host__ __device__ inline vec3 operator-() const {
    return vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ inline float operator[](int i) const { return e[i]; }
  __host__ __device__ inline float& operator[](int i) { return e[i]; }

  __host__ __device__ inline vec3& operator+=(const vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
  }

  __host__ __device__ inline vec3& operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
  }
  __host__ __device__ inline vec3& operator*=(const vec3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
  }
  __host__ __device__ inline vec3& operator/=(const vec3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
  }
  __host__ __device__ inline vec3& operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
  }
  __host__ __device__ inline vec3& operator/=(const float t) {
    return *this *= (1 / t);
  }

  __host__ __device__ inline float norm_sqr() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }

  __host__ __device__ inline float norm() const {
    return std::sqrt(norm_sqr());
  }

  __host__ __device__ bool near_null() const {
    const float s = 1e-8;
    return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) &&
           (std::fabs(e[2]) < s);
  }
  __host__ __device__ inline void make_unit_vector();

 private:
  float e[3];
};

// Can I get away without using this?
// I did before
// FIXME: THIS IS NOT RIGHT, TOO LAZY TO DO RIGHT NOW!!!
inline std::istream& operator>>(std::istream& is, vec3& v) {
  // is >> v.setX() >> v.setY() >> v.setZ();
  v.setX(is);
  v.setY(is);
  v.setZ(is);
  return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& v) {
  os << v.getX() << " " << v.getY() << " " << v.getZ();
  return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
  float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
  return vec3(u.getX() + v.getX(), u.getY() + v.getY(), u.getZ() + v.getZ());
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
  return vec3(u.getX() - v.getX(), u.getY() - v.getY(), u.getZ() - v.getZ());
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
  return vec3(u.getX() * v.getX(), u.getY() * v.getY(), u.getZ() * v.getZ());
}

__host__ __device__ inline vec3 operator/(const vec3& u, const vec3& v) {
  return vec3(u.getX() / v.getX(), u.getY() / v.getY(), u.getZ() / v.getZ());
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
  return vec3(t * v.getX(), t * v.getY(), t * v.getZ());
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
  return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
  return (1.0f / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
  return u.getX() * v.getX() + u.getY() * v.getY() + u.getZ() * v.getZ();
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
  return vec3((u.getY() * v.getZ() - u.getZ() * v.getY()),
              (-(u.getX() * v.getZ() - u.getZ() * v.getX())),
              (u.getX() * v.getY() - u.getY() * v.getX()));
}

__host__ __device__ inline vec3 unit_v(vec3 v) {
  return v / v.norm();
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
  return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3& uv,
                                        const vec3& n,
                                        float etai_over_etat) {
  float cos_theta = std::fmin(dot(-uv, n), 1.0f);
  vec3 r_out_tang = etai_over_etat * (uv + cos_theta * n);
  vec3 r_out_parr = (-std::sqrt(std::fabs(1.0f - r_out_tang.norm_sqr()))) * n;
  return r_out_parr + r_out_tang;
}

__device__ inline bool refract(const vec3& v,
                               const vec3& n,
                               float etai_over_etat,
                               vec3& refr) {
  vec3 uv = unit_v(v);
  float dt = dot(uv, n);
  float d = 1.0f - etai_over_etat * etai_over_etat * (1 - dt * dt);
  if (d > 0) {
    refr = etai_over_etat * (uv - n * dt) - n * std::sqrt(d);
    return true;
  } else {
    return false;
  }
}

__device__ inline vec3 rand_unit_sphere(curandState* local_rand_state) {
  vec3 p;
  do {
    p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
  } while (p.norm_sqr() >= 1.0f);
  return p;
}

__device__ vec3 rand_unit_disk(curandState* local_rand_state) {
  vec3 p;
  do {
    p = 2.0f * vec3(curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state), 0) -
        vec3(1, 1, 0);
  } while (dot(p, p) >= 1.0f);
  return p;
}

using point3 = vec3;
using color = vec3;

#endif  // INCLUDE_VEC3_HPP_
