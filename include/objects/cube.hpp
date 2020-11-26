#ifndef INCLUDE_OBJECTS_CUBE_HPP_
#define INCLUDE_OBJECTS_CUBE_HPP_

#include "bounding_box.hpp"
#include "hit_list.hpp"
#include "rectangle.hpp"
#include "vec3.hpp"

class cube : public hit {
 public:
  __device__ cube();
  __device__ cube(const point3&, const point3&, material* m);

  __device__ bool is_hit(const ray& r,
                         float t_min,
                         float t_max,
                         hit_rec& rec) const override {
    return sides_l.is_hit(r, t_min, t_max, rec);
  }

  __device__ bool bound_box(float, float, BB& out) const override {
    out = BB(min_, max_);
    return true;
  }

 private:
  point3 min_;
  point3 max_;
  hit* list[6];
  hit_list sides_l;
};

__device__ cube::cube(const point3& min, const point3& max, material* m) {
  min_ = min;
  max_ = max;
  list[0] = new xy_rectangle(min.getX(), max.getX(), min.getY(), max.getY(),
                             max.getZ(), m);
  list[1] = new xy_rectangle(min.getX(), max.getX(), min.getY(), max.getY(),
                             min.getZ(), m);
  list[2] = new xz_rectangle(min.getX(), max.getX(), min.getZ(), max.getZ(),
                             max.getY(), m);
  list[3] = new xz_rectangle(min.getX(), max.getX(), min.getZ(), max.getZ(),
                             min.getY(), m);
  list[4] = new yz_rectangle(min.getY(), max.getY(), min.getZ(), max.getZ(),
                             max.getX(), m);
  list[5] = new yz_rectangle(min.getY(), max.getY(), min.getZ(), max.getZ(),
                             min.getX(), m);
  sides_l = hit_list(list, 6);
}

#endif  // INCLUDE_OBJECTS_CUBE_HPP_
