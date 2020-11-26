#ifndef INCLUDE_OBJECTS_HIT_LIST_HPP_
#define INCLUDE_OBJECTS_HIT_LIST_HPP_

#include "bounding_box.hpp"
#include "hit.hpp"

class hit_list : public hit {
 public:
  __device__ hit_list() {}
  __device__ hit_list(hit** l, int n) : l_(l), n_(n) {}
  __device__ bool is_hit(const ray&, float, float, hit_rec&) const override;
  __device__ bool bound_box(float, float, BB&) const override;

  __device__ inline float size() const { return n_; }
  __device__ inline hit** objects() const { return l_; }

 private:
  hit** l_;
  int n_;
};

__device__ bool hit_list::is_hit(const ray& r,
                                 float t_min,
                                 float t_max,
                                 hit_rec& rec) const {
  hit_rec temp;
  bool hit_ = false;
  float best_guess = t_max;

  for (int i = 0; i < n_; i++) {
    if (l_[i]->is_hit(r, t_min, best_guess, temp)) {
      hit_ = true;
      best_guess = temp.t;
      rec = temp;
    }
  }
  return hit_;
}

__device__ bool hit_list::bound_box(float t0, float t1, BB& out) const {
  if (l_ == NULL)
    return false;

  BB temp;
  bool first = true;

  // No idea if this will work :/
  for (int i = 0; i < n_; i++) {
    if (l_[i]->bound_box(t0, t1, temp))
      return false;
    out = first ? temp : surround_box(out, temp);
    first = false;
  }

  return true;
}

#endif  // INCLUDE_OBJECTS_HIT_LIST_HPP_
