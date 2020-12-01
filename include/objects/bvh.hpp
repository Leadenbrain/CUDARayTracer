#ifndef INCLUDE_OBJECTS_BVH_HPP_
#define INCLUDE_OBJECTS_BVH_HPP_

#include <algorithm>
#include <cmath>
#include <vector>
#include "utilities.hpp"

#include "bounding_box.hpp"
#include "hit.hpp"
#include "hit_list.hpp"

class bvh_node : public hit {
 public:
  __device__ bvh_node() {}
  __device__ bvh_node(hit_list* l,
                      float t0,
                      float t1,
                      curandState* local_rand_state)
      : bvh_node(l->objects(), 0, l->size(), t0, t1, local_rand_state) {}
  __device__ bvh_node(hit**, size_t, size_t, float, float, curandState*);

  __device__ bool bound_box(float, float, BB& out) const override {
    out = box;
    return true;
  }

  __device__ bool is_hit(const ray& r,
                         float t_min,
                         float t_max,
                         hit_rec& rec) const override {
    if (!box.is_hit(r, t_min, t_max))
      return false;

    bool hit_l = left_->is_hit(r, t_min, t_max, rec);
    bool hit_r = right_->is_hit(r, t_min, hit_l ? rec.t : t_max, rec);

    return hit_l || hit_r;
  }

 private:
  hit* left_;
  hit* right_;
  BB box;
};

__device__ inline bool comp(const hit* a, const hit* b, const int& axis) {
  BB b1;
  BB b2;

  if (!a->bound_box(0, 0, b1) || !b->bound_box(0, 0, b2))
    return false;
  // std::cerr << "No box in bvh_node constructor \n";

  if (axis == 0)
    return b1.min().getX() < b2.min().getX();
  else if (axis == 1)
    return b1.min().getY() < b2.min().getY();
  else
    return b1.min().getZ() < b2.min().getZ();
}

__device__ inline bool x_comp(const hit* a, const hit* b) {
  return comp(a, b, 0);
}

__device__ inline bool y_comp(const hit* a, const hit* b) {
  return comp(a, b, 1);
}

__device__ inline bool z_comp(const hit* a, const hit* b) {
  return comp(a, b, 2);
}

__device__ bvh_node::bvh_node(hit** src,
                              size_t start,
                              size_t end,
                              float t0,
                              float t1,
                              curandState* local_rand_state) {
  // FIXME: NEED OBJ = src
  auto obj = src;
  float rand = curand_uniform(local_rand_state);
  rand *= (2.999999f);
  int axis = (int)truncf(rand);

  auto comparator = (axis == 0) ? x_comp : (axis == 1) ? y_comp : z_comp;

  size_t obj_len = end - start;
  if (obj_len == 1) {
    left_ = right_ = obj[start];
  } else if (obj_len == 2) {
    if (comparator(obj[start], obj[start + 1])) {
      left_ = obj[start];
      right_ = obj[start + 1];
    } else {
      left_ = obj[start + 1];
      right_ = obj[start];
    }
  } else {
    // std::sort(obj->begin() + start, obj->begin() + end, comparator);
    // printf("HERE\n");
    size_t mid = start + obj_len / 2;
    left_ = new bvh_node(obj, start, mid, t0, t1, local_rand_state);
    right_ = new bvh_node(obj, mid, end, t0, t1, local_rand_state);
  }

  BB bL, bR;

  if (!left_->bound_box(t0, t1, bL) || !right_->bound_box(t0, t1, bR))
    printf("ERROR NO BOX IN BVH NODE CONSTRUCTOR");
  box = surround_box(bL, bR);
}

#endif  // INCLUDE_OBJCETS_BVH_HPP_
