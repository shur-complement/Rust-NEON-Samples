// Example - collision detection
// Original in C++ at https://developer.arm.com/documentation/102467/0201/Example---collision-detection

use std::arch::aarch64::*;

struct Circle {
    x: f32,
    y: f32,
    radius: f32,
}

struct Circles {
    size: usize,
    xs: Vec<f32>,
    ys: Vec<f32>,
    radii: Vec<f32>,
}

fn does_collide_neon_deinterleaved(
    input: &Circles,
    collider: &Circle,
    out: &mut [u32],
) {
unsafe {

  // Duplicate the collider properties in 3 separate 4-lane vector registers
  let c1_x = vdupq_n_f32(collider.x);
  let c1_y = vdupq_n_f32(collider.y);
  let c1_r = vdupq_n_f32(collider.radius);

  let xs_ptr = input.xs.as_ptr();
  let ys_ptr = input.ys.as_ptr();
  let rs_ptr = input.radii.as_ptr();

  // Perform 4 collision tests at a time
  for offset in (0..input.size).step_by(4) {

    let x                   = vld1q_f32(xs_ptr.add(offset));
    let y                   = vld1q_f32(ys_ptr.add(offset));

    let delta_x             = vsubq_f32(c1_x, x);
    let delta_y             = vsubq_f32(c1_y, y);

    let delta_x_squared     = vmulq_f32(delta_x, delta_x);
    let delta_y_squared     = vmulq_f32(delta_y, delta_y);

    let sum_deltas_squared  = vaddq_f32(delta_x_squared, delta_y_squared);

    let r = vld1q_f32(rs_ptr.add(offset));
    let radius_sum          = vaddq_f32(c1_r, r);
    let radius_sum_squared  = vmulq_f32(radius_sum, radius_sum);

    let mask                = vcltq_f32(sum_deltas_squared, radius_sum_squared);

    // Unpack the results in each lane
    out[offset]             = 1 & vgetq_lane_u32(mask, 0);
    out[offset + 1]         = 1 & vgetq_lane_u32(mask, 1);
    out[offset + 2]         = 1 & vgetq_lane_u32(mask, 2);
    out[offset + 3]         = 1 & vgetq_lane_u32(mask, 3);
  }
}
}

fn main() {

  const NUM_INPUT: usize = 4;

  let mut input_x = vec![0.; NUM_INPUT];
  let mut input_y = vec![0.; NUM_INPUT];
  let mut input_r = vec![0.; NUM_INPUT];
  let mut output = vec![0; NUM_INPUT];

  // Set up the data for multiple circles
  for i in 0..NUM_INPUT {
    input_x[i] = (i*2) as f32;
    input_y[i] = (i*3) as f32;
    input_r[i] = i as f32;
    output[i] = 0;
  }

  // Organize objects in Struct of Arrays layout
  let c1 = Circles {
    size: NUM_INPUT,
    radii: input_r,
    xs: input_x,
    ys: input_y,
  };

  // collider object
  let c2 = Circle {
      radius: 5.0,
      x: 10.0,
      y: 10.0,
  };

  // Test whether the collider circle collides with any of the input circles
  does_collide_neon_deinterleaved(&c1, &c2, &mut output);

  // Iterate over the returned output data and display results
  for i in 0..NUM_INPUT {
    if output[i] != 0 {
      println!("Circle {} at ({}, {}) with radius {} collides", i, c1.xs[i], c1.ys[i], c1.radii[i]);
    } else {
    println!("Circle {} at ({}, {}) with radius {} does not collide", i, c1.xs[i], c1.ys[i], c1.radii[i]);
    }
  }
}
