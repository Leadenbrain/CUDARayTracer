#ifndef INCLUDE_MATERIALS_GLASS_HPP_
#define INCLUDE_MATERIALS_GLASS_HPP_

#include "material.hpp"

class glass : public material {
 public:
  explicit __device__ glass(const float& index) : eta_(index) {}
  __device__ virtual bool scatter(
      const ray& r,
      const hit_rec& rec,
      color& att,
      ray& scat,
      curandState* local_rand_state) const override {
#ifdef MY_GLASS
    float refr_rat = rec.front ? (1.0 / eta_) : eta_;

    vec3 unit_d = unit_v(r.direction());

    float cos_t = std::fmin(dot(-unit_d, rec.n), 1.0f);
    float sin_t = std::sqrt(1.0f - cos_t * cos_t);

    bool refl = refr_rat * sin_t > 1.0;
    vec3 direction;

    if (refl || schlick(cos_t, refr_rat) > curand_uniform(local_rand_state))
      direction = reflect(unit_d, rec.n);
    else
      direction = refract(unit_d, rec.n, refr_rat);

    scat = ray(rec.p, direction, r.time());
    att = color(1.0f, 1.0f, 1.0f);
    return true;
#else
    vec3 n_out;
    vec3 refl = reflect(r.direction(), rec.n);
    float etai_over_etat;
    att = vec3(1.0f, 1.0f, 1.0f);
    vec3 refr;
    float refl_prob;
    float cos;
    if (dot(r.direction(), rec.n) > 0.0f) {
      n_out = -rec.n;
      etai_over_etat = eta_;
      cos = dot(r.direction(), rec.n) / r.direction().norm();
      cos = std::sqrt(1.0f - eta_ * eta_ * (1 - cos * cos));
    } else {
      n_out = rec.n;
      etai_over_etat = 1.0f / eta_;
      cos = -dot(r.direction(), rec.n) / r.direction().norm();
    }
    if (refract(r.direction(), n_out, etai_over_etat, refr))
      refl_prob = schlick(cos, eta_);
    else
      refl_prob = 1.0f;
    if (curand_uniform(local_rand_state) < refl_prob)
      scat = ray(rec.p, refl, r.time());
    else
      scat = ray(rec.p, refr, r.time());
    return true;
#endif  // FAST_GLASS
  }

 private:
  float eta_;

  __device__ float schlick(float cos, float eta) const {
    float eta0 = (1.0f - eta) / (1.0f + eta);
    eta0 *= eta0;
    return eta0 + (1.0f - eta0) * std::pow((1.0f - cos), 5.0f);
  }
};

#endif  // INCLUDE_MATERIALS_GLASS_HPP_
